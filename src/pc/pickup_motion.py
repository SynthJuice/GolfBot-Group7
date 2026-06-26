import math
import time

from collection_algorithm import A_star
from com_protocol import build_goto, send_command
from map_utils import (
    clamp_map_point,
    heading_from_map_points,
    map_point_is_valid,
    path_is_valid,
    point_distance,
    robot_center_point,
)
from path_obstacles import (
    clear_path_endpoint,
    clear_path_endpoint_preserving_obstacles,
    choose_safe_path_lookahead,
    clone_path_matrix,
    mark_red_cross_obstacles,
    point_in_obstacle_regions,
    red_cross_obstacle_regions,
    segment_intersects_regions,
)
from pickup_scene import capture_pickup_scene, planning_matrix_from_scene
from robot_sync import (
    back_off_from_red_cross,
    goto_map_point_with_pose_pre_turn,
    map_xy_to_ev3_xy,
    normalize_turn_angle,
    reverse_for_missing_grappler,
    reverse_for_missing_robot,
    sync_robot_from_camera,
    sync_robot_pose_value,
    turn_robot_to_heading,
)
from scene_analysis import robot_body_visible
from settings import (
    MAP_HEIGHT,
    MAP_WIDTH,
    PICKUP_BALL_ENDPOINT_CLEAR_RADIUS,
    PICKUP_CENTER_TO_BALL_CLOSE_DISTANCE,
    PICKUP_CENTER_TO_BALL_MARGIN,
    PICKUP_FINAL_HEADING_CLOSE_TOLERANCE,
    PICKUP_FINAL_SCOOP_DISTANCE,
    PICKUP_GRAPPLER_CLOSE_DISTANCE,
    PICKUP_OFFCENTER_DISTANCE_SCALE,
    PICKUP_OFFCENTER_SCOOP_SCALE_LIMIT,
    PICKUP_RED_CROSS_CLEARANCE_MARGIN,
    PICKUP_RED_CROSS_LOOKAHEAD_DISTANCE,
    PICKUP_SERVO_FAR_FORWARD_STEP,
    PICKUP_SERVO_MAX_FORWARD_STEP,
    PICKUP_SERVO_MAX_ITERATIONS,
    PICKUP_SERVO_MID_FORWARD_STEP,
    PICKUP_SERVO_MIN_FORWARD_STEP,
    PICKUP_SERVO_NEAR_FORWARD_STEP,
    RED_CROSS_BACKOFF_MAX_ATTEMPTS,
    RED_CROSS_WAYPOINT_ACCEPTANCE_RADIUS,
    SYNC_DELAY_SECONDS,
)
from vision_detection import set_vision_path_overlay


def path_point_at_distance(robot_path, target_distance):
    if not path_is_valid(robot_path):
        return None

    if len(robot_path) == 1:
        return robot_path[0]

    target_distance = max(0.0, float(target_distance))
    travelled = 0.0

    for index in range(1, len(robot_path)):
        previous_point = robot_path[index - 1]
        current_point = robot_path[index]
        segment_length = point_distance(previous_point, current_point)

        if travelled + segment_length >= target_distance:
            if segment_length <= 0.001:
                return current_point

            ratio = (target_distance - travelled) / segment_length
            row = previous_point[0] + (current_point[0] - previous_point[0]) * ratio
            col = previous_point[1] + (current_point[1] - previous_point[1]) * ratio
            return int(round(row)), int(round(col))

        travelled += segment_length

    return robot_path[-1]


def red_cross_routed_pickup_step(scene, start_point, target_point, distance_map_units):
    if scene is None:
        return None

    base_matrix = planning_matrix_from_scene(scene)
    vision_scene = scene.get("vision_scene")
    regions = red_cross_obstacle_regions(
        base_matrix,
        vision_scene,
        margin=PICKUP_RED_CROSS_CLEARANCE_MARGIN,
    )

    if not segment_intersects_regions(start_point, target_point, regions):
        return None

    path_matrix = clone_path_matrix(base_matrix)
    mark_red_cross_obstacles(
        path_matrix,
        vision_scene,
        margin=PICKUP_RED_CROSS_CLEARANCE_MARGIN,
    )
    clear_path_endpoint(path_matrix, start_point, radius=10, value=".")
    clear_path_endpoint_preserving_obstacles(
        path_matrix,
        path_matrix,
        target_point,
        radius=PICKUP_BALL_ENDPOINT_CLEAR_RADIUS,
        value=".",
        blocked_regions=regions,
    )
    robot_path = A_star(path_matrix, start_point, target_point)

    if not path_is_valid(robot_path):
        print(
            "Pickup servo: red cross blocks direct step, but A* could not find a safe route: {}".format(
                robot_path,
            )
        )
        return False

    lookahead_distance = max(
        float(distance_map_units),
        float(PICKUP_RED_CROSS_LOOKAHEAD_DISTANCE),
    )
    step_target = choose_safe_path_lookahead(
        robot_path,
        start_point,
        regions,
        min_distance=max(
            float(distance_map_units),
            float(RED_CROSS_WAYPOINT_ACCEPTANCE_RADIUS),
        ),
        max_distance=lookahead_distance,
        acceptance_radius=RED_CROSS_WAYPOINT_ACCEPTANCE_RADIUS,
    )

    if step_target is None:
        step_target = path_point_at_distance(robot_path, distance_map_units)

        if (
            step_target is None
            or point_in_obstacle_regions(step_target, regions)
            or segment_intersects_regions(start_point, step_target, regions)
        ):
            print("Pickup servo: red cross route had no safe lookahead waypoint")
            return False

    set_vision_path_overlay(
        [
            {
                "points": robot_path,
                "label": "Pickup red-cross route",
                "color": (255, 0, 255),
            },
            {
                "points": [start_point, step_target],
                "label": "Pickup lookahead",
                "color": (0, 255, 255),
            },
        ],
        label="Pickup route",
    )

    print(
        "Pickup servo: red cross blocks direct step; routing lookahead via {} "
        "(lookahead {:.1f})".format(
            step_target,
            lookahead_distance,
        )
    )
    return step_target


def pickup_offcenter_ratio(point):
    if point is None:
        return 0.0

    row, col = point
    center_row = float(MAP_HEIGHT - 1) / 2.0
    center_col = float(MAP_WIDTH - 1) / 2.0
    max_distance = math.hypot(center_row, center_col)

    if max_distance <= 0.0:
        return 0.0

    distance = math.hypot(float(row) - center_row, float(col) - center_col)
    return max(0.0, min(1.0, distance / max_distance))


def pickup_distance_scale(point):
    return 1.0 + float(PICKUP_OFFCENTER_DISTANCE_SCALE) * pickup_offcenter_ratio(point)


def scaled_pickup_distance(distance, point):
    scale = pickup_distance_scale(point)
    return float(distance) * scale, scale


def scaled_final_scoop_distance(point):
    scale = min(
        float(PICKUP_OFFCENTER_SCOOP_SCALE_LIMIT),
        pickup_distance_scale(point),
    )
    return float(PICKUP_FINAL_SCOOP_DISTANCE) * scale


def pickup_servo_forward_step(center_to_ball):
    if center_to_ball > 140.0:
        return min(PICKUP_SERVO_MAX_FORWARD_STEP, PICKUP_SERVO_FAR_FORWARD_STEP)

    if center_to_ball > 80.0:
        return min(PICKUP_SERVO_MAX_FORWARD_STEP, PICKUP_SERVO_MID_FORWARD_STEP)

    return min(PICKUP_SERVO_MAX_FORWARD_STEP, PICKUP_SERVO_NEAR_FORWARD_STEP)


def drive_toward_map_point(sock, camera, robot_pose, target_point, distance_map_units, scene=None):
    center_x, center_y, _heading = robot_pose
    center_point = (int(round(center_y)), int(round(center_x)))
    distance_to_target = point_distance(center_point, target_point)

    if distance_to_target <= 0.001:
        print("Pickup servo: center is already at target point")
        return True

    distance_map_units = float(distance_map_units)
    routed_step = red_cross_routed_pickup_step(
        scene,
        center_point,
        target_point,
        distance_map_units,
    )

    if routed_step is False:
        regions = red_cross_obstacle_regions(
            planning_matrix_from_scene(scene),
            scene.get("vision_scene") if scene is not None else None,
            margin=PICKUP_RED_CROSS_CLEARANCE_MARGIN,
        )

        if back_off_from_red_cross(
            sock,
            robot_pose=robot_pose,
            regions=regions,
            label="Pickup servo route recovery",
        ):
            return True

        return False

    if routed_step is not None:
        target_row, target_col = routed_step
    else:
        unit_row = (float(target_point[0]) - float(center_point[0])) / distance_to_target
        unit_col = (float(target_point[1]) - float(center_point[1])) / distance_to_target

        target_row = int(round(float(center_point[0]) + distance_map_units * unit_row))
        target_col = int(round(float(center_point[1]) + distance_map_units * unit_col))

    if not map_point_is_valid((target_row, target_col)):
        clamped_row, clamped_col = clamp_map_point((target_row, target_col), margin=5)
        print(
            "Pickup servo center target was out of bounds: {}; "
            "using clamped target ({}, {})".format(
                (target_row, target_col),
                clamped_col,
                clamped_row,
            )
        )
        target_row, target_col = clamped_row, clamped_col

    if routed_step is not None:
        return goto_map_point_with_pose_pre_turn(
            sock,
            camera,
            robot_pose,
            (target_row, target_col),
            label="Pickup red-cross routed step",
        )

    print(
        "Pickup servo: center->ball distance={:.1f}; moving {:.1f} to map center=({}, {})".format(
            distance_to_target,
            distance_map_units,
            target_col,
            target_row,
        )
    )
    set_vision_path_overlay(
        [center_point, (target_row, target_col), target_point],
        label="Pickup servo step",
        color=(0, 255, 255),
    )

    if not sync_robot_pose_value(sock, robot_pose, label="Pickup servo pre-GOTO"):
        return False

    ev3_x, ev3_y = map_xy_to_ev3_xy(target_col, target_row)
    print("Pickup servo: GOTO ev3=({}, {})".format(ev3_x, ev3_y))

    if not send_command(sock, build_goto(ev3_x, ev3_y)):
        return False

    time.sleep(SYNC_DELAY_SECONDS)
    return sync_robot_from_camera(sock, camera)


def final_scoop_is_safe_from_red_cross(scene, robot_pose, target_point):
    if scene is None:
        return True

    base_matrix = planning_matrix_from_scene(scene)
    vision_scene = scene.get("vision_scene")
    regions = red_cross_obstacle_regions(
        base_matrix,
        vision_scene,
        margin=PICKUP_RED_CROSS_CLEARANCE_MARGIN,
    )

    if not regions:
        return True

    center_x, center_y, _heading = robot_pose
    center_point = (int(round(center_y)), int(round(center_x)))
    grappler_point = scene.get("grappler_point")
    ball_point = scene.get("ball_point")

    for label, point in (
        ("robot center", center_point),
        ("grappler", grappler_point),
        ("ball", ball_point),
    ):
        if point_in_obstacle_regions(point, regions):
            print(
                "Pickup final scoop: refusing to move because {} {} is inside the red-cross pickup clearance".format(
                    label,
                    point,
                )
            )
            return False

    if grappler_point is not None and ball_point is not None:
        if segment_intersects_regions(grappler_point, ball_point, regions):
            print(
                "Pickup final scoop: refusing to close because the grappler-to-ball line crosses the red-cross pickup clearance"
            )
            return False

    if segment_intersects_regions(center_point, target_point, regions):
        print(
            "Pickup final scoop: refusing forward move because it crosses the red-cross pickup clearance"
        )
        return False

    return True


def final_scoop_forward_before_close(
    sock,
    camera,
    robot_pose,
    distance_map_units=PICKUP_FINAL_SCOOP_DISTANCE,
    scene=None,
):
    if distance_map_units <= 0.0:
        return True

    center_x, center_y, heading = robot_pose
    heading_rad = math.radians(float(heading))
    target_x = int(round(float(center_x) + float(distance_map_units) * math.cos(heading_rad)))
    target_y = int(round(float(center_y) + float(distance_map_units) * math.sin(heading_rad)))

    target_row, target_col = clamp_map_point((target_y, target_x), margin=5)
    target_x = target_col
    target_y = target_row
    target_point = (target_y, target_x)

    if not final_scoop_is_safe_from_red_cross(scene, robot_pose, target_point):
        regions = red_cross_obstacle_regions(
            planning_matrix_from_scene(scene),
            scene.get("vision_scene") if scene is not None else None,
            margin=PICKUP_RED_CROSS_CLEARANCE_MARGIN,
        )
        back_off_from_red_cross(
            sock,
            robot_pose=robot_pose,
            regions=regions,
            label="Pickup final scoop recovery",
        )
        return False

    print(
        "Pickup final scoop: moving forward {:.1f} map units to map center=({}, {}) before closing".format(
            distance_map_units,
            target_x,
            target_y,
        )
    )
    set_vision_path_overlay(
        [robot_center_point(robot_pose), target_point],
        label="Pickup final scoop",
        color=(0, 255, 255),
    )

    if not sync_robot_pose_value(sock, robot_pose, label="Pickup final scoop pre-GOTO"):
        return False

    ev3_x, ev3_y = map_xy_to_ev3_xy(target_x, target_y)
    print("Pickup final scoop: GOTO ev3=({}, {})".format(ev3_x, ev3_y))

    if not send_command(sock, build_goto(ev3_x, ev3_y)):
        return False

    time.sleep(SYNC_DELAY_SECONDS)
    return sync_robot_from_camera(sock, camera)


def servo_align_and_approach_ball(sock, camera, ball_color="W", target_ball_point=None):
    last_scene = None
    red_cross_backoff_count = 0

    for iteration in range(1, PICKUP_SERVO_MAX_ITERATIONS + 1):
        scene = capture_pickup_scene(
            camera,
            ball_color=ball_color,
            target_point=target_ball_point,
        )
        last_scene = scene

        if scene is None:
            print("Pickup servo: could not read camera frame")
            return False

        robot_pose = scene["robot_pose"]
        grappler_point = scene["grappler_point"]
        ball_point = scene["ball_point"]

        if grappler_point is None:
            if robot_body_visible(scene["vision_scene"]):
                if reverse_for_missing_grappler(sock, label="Pickup servo"):
                    continue
                return False

            print("Pickup servo: missing grappler and robot body detection")
            return False

        if robot_pose is None:
            print("Pickup servo: missing robot detection")
            if reverse_for_missing_robot(sock, label="Pickup servo"):
                continue
            return False

        red_cross_regions = red_cross_obstacle_regions(
            planning_matrix_from_scene(scene),
            scene.get("vision_scene"),
            margin=PICKUP_RED_CROSS_CLEARANCE_MARGIN,
        )
        servo_center_point = (int(round(robot_pose[1])), int(round(robot_pose[0])))
        robot_inside_clearance = point_in_obstacle_regions(servo_center_point, red_cross_regions)
        grappler_inside_clearance = point_in_obstacle_regions(grappler_point, red_cross_regions)

        if robot_inside_clearance or grappler_inside_clearance:
            if red_cross_backoff_count >= RED_CROSS_BACKOFF_MAX_ATTEMPTS:
                print(
                    "Pickup servo: still inside the red-cross clearance after {} backoff(s); "
                    "continuing escape instead of stopping".format(red_cross_backoff_count)
                )

            blocked_part = "robot center" if robot_inside_clearance else "claw"
            print(
                "Pickup servo: {} is inside the red-cross clearance; backing away "
                "instead of stopping (backoff {}/{})".format(
                    blocked_part,
                    red_cross_backoff_count + 1,
                    RED_CROSS_BACKOFF_MAX_ATTEMPTS,
                )
            )

            if not back_off_from_red_cross(
                sock,
                robot_pose=robot_pose,
                regions=red_cross_regions,
                label="Pickup servo",
            ):
                return False

            red_cross_backoff_count += 1
            continue

        red_cross_backoff_count = 0

        if ball_point is None:
            if scene.get("pickup_target_missing"):
                print(
                    "Pickup servo: selected ball target is not visible; retrying instead "
                    "of switching to a different nearby ball"
                )
                return False

            if scene.get("pickup_blocked_by_red_cross"):
                print(
                    "Pickup servo: only visible ball is inside the red-cross pickup clearance; not closing claw"
                )
                return False

            print(
                "Pickup servo: ball is no longer visible; assuming it is at/inside the claw and closing"
            )
            return True

        center_x, center_y, current_heading = robot_pose
        center_point = (int(round(center_y)), int(round(center_x)))
        center_to_ball_raw = point_distance(center_point, ball_point)
        center_to_ball, perspective_scale = scaled_pickup_distance(center_to_ball_raw, ball_point)
        final_scoop_distance = scaled_final_scoop_distance(ball_point)
        target_heading = heading_from_map_points(center_point, ball_point)
        heading_error = normalize_turn_angle(target_heading - current_heading)

        if grappler_point is not None:
            grappler_to_ball_raw = point_distance(grappler_point, ball_point)
            grappler_to_ball = grappler_to_ball_raw * perspective_scale
            grappler_text = (
                ", grappler={}, grappler_distance={:.1f}, "
                "scaled_grappler_distance={:.1f}"
            ).format(
                grappler_point,
                grappler_to_ball_raw,
                grappler_to_ball,
            )
        else:
            grappler_text = ", grappler=None"

        print(
            "Pickup servo iteration {}: center={}, ball={}, center_distance={:.1f}, "
            "scaled_center_distance={:.1f}, perspective_scale={:.2f}, "
            "heading={:.1f}, target_heading={:.1f}, heading_error={:.1f}{}".format(
                iteration,
                center_point,
                ball_point,
                center_to_ball_raw,
                center_to_ball,
                perspective_scale,
                current_heading,
                target_heading,
                heading_error,
                grappler_text,
            )
        )

        if grappler_point is not None and grappler_to_ball <= PICKUP_GRAPPLER_CLOSE_DISTANCE:
            print(
                "Pickup servo: ball is inside grappler range; doing final scoop instead of turning"
            )
            return final_scoop_forward_before_close(
                sock,
                camera,
                robot_pose,
                distance_map_units=final_scoop_distance,
                scene=scene,
            )

        if center_to_ball <= PICKUP_CENTER_TO_BALL_CLOSE_DISTANCE + PICKUP_CENTER_TO_BALL_MARGIN:
            if abs(heading_error) > PICKUP_FINAL_HEADING_CLOSE_TOLERANCE:
                print("Pickup servo: close to ball, doing one final heading correction")
                if not turn_robot_to_heading(
                    sock,
                    camera,
                    target_heading,
                    tolerance_degrees=6.0,
                ):
                    return False

                continue

            print("Pickup servo: center is close enough; doing final scoop before claw close")
            return final_scoop_forward_before_close(
                sock,
                camera,
                robot_pose,
                distance_map_units=final_scoop_distance,
                scene=scene,
            )

        max_step = pickup_servo_forward_step(center_to_ball)
        forward_distance = center_to_ball - PICKUP_CENTER_TO_BALL_CLOSE_DISTANCE
        forward_distance = min(forward_distance, max_step)

        if forward_distance < PICKUP_SERVO_MIN_FORWARD_STEP:
            print("Pickup servo: remaining center move is tiny; doing final scoop before claw close")
            return final_scoop_forward_before_close(
                sock,
                camera,
                robot_pose,
                distance_map_units=final_scoop_distance,
                scene=scene,
            )

        print(
            "Pickup servo step choice: max_step={:.1f}, requested_forward={:.1f}".format(
                max_step,
                forward_distance,
            )
        )

        if not drive_toward_map_point(
            sock,
            camera,
            robot_pose,
            ball_point,
            forward_distance,
            scene=scene,
        ):
            return False

    final_scene = capture_pickup_scene(
        camera,
        ball_color=ball_color,
        target_point=target_ball_point,
    )
    if final_scene is not None:
        if final_scene.get("pickup_target_missing"):
            print(
                "Pickup servo: final frame lost the selected ball target; not closing claw on a different ball"
            )
            return False

        if final_scene.get("pickup_blocked_by_red_cross"):
            print(
                "Pickup servo: final frame only sees a ball inside the red-cross pickup clearance; not closing claw"
            )
            return False

        final_robot = final_scene["robot_pose"]
        final_grappler = final_scene["grappler_point"]
        final_ball = final_scene["ball_point"]

        if final_grappler is None and robot_body_visible(final_scene["vision_scene"]):
            reverse_for_missing_grappler(sock, label="Pickup final frame")
            return False

        if final_robot is not None and final_ball is not None:
            final_center = (int(round(final_robot[1])), int(round(final_robot[0])))
            final_distance_raw = point_distance(final_center, final_ball)
            final_distance, final_scale = scaled_pickup_distance(final_distance_raw, final_ball)
            final_scoop_distance = scaled_final_scoop_distance(final_ball)
            print(
                "Pickup servo: max iterations reached; final center distance={:.1f}, "
                "scaled_final_distance={:.1f}, perspective_scale={:.2f}".format(
                    final_distance_raw,
                    final_distance,
                    final_scale,
                )
            )
            if final_distance > PICKUP_CENTER_TO_BALL_CLOSE_DISTANCE + 8.0:
                print("Pickup servo: still too far from ball; not closing claw")
                return False

            if final_robot is not None:
                return final_scoop_forward_before_close(
                    sock,
                    camera,
                    final_robot,
                    distance_map_units=final_scoop_distance,
                    scene=final_scene,
                )

    print("Pickup servo: max iterations reached but final distance is acceptable; closing")
    return True
