import math
import time

from collection_algorithm import A_star
from com_protocol import (
    build_claw_close,
    build_claw_deliver,
    build_setspeed,
    build_turn,
    send_command,
)
from map_utils import (
    clamp_map_point,
    path_is_valid,
    point_distance,
    robot_center_point,
    simplify_path_for_robot,
)
from path_obstacles import (
    clear_path_endpoint,
    choose_safe_path_lookahead,
    clone_path_matrix,
    create_empty_path_matrix,
    mark_red_cross_obstacles,
    point_in_obstacle_regions,
    red_cross_obstacle_regions,
    segment_intersects_regions,
)
from robot_sync import (
    back_off_from_red_cross,
    goto_then_sync_with_pre_turn,
    goto_map_point_with_pose,
    normalize_turn_angle,
    reverse_for_missing_grappler,
    reverse_for_missing_robot,
    sync_robot_pose_from_camera,
    sync_robot_pose_value,
)
from scene_analysis import (
    capture_scene_with_robot_pose_retry,
    capture_vision_scene_frame,
    robot_body_visible,
    robot_pose_from_sources,
)
from settings import (
    DELIVERY_CENTER_TO_MARKER_DISTANCE,
    DELIVERY_CENTER_TO_CLAW_DISTANCE,
    DELIVERY_CLAW_EDGE_MARGIN,
    DELIVERY_CLAW_POSITION_TOLERANCE,
    DELIVERY_CLAW_TO_MARKER_DISTANCE,
    DELIVERY_DYNAMIC_REPLAN_MAX_STEPS,
    DELIVERY_EDGE_ESCAPE_REVERSE_SPEED,
    DELIVERY_EDGE_ESCAPE_SECONDS,
    DELIVERY_FINAL_CORRECTION_ATTEMPTS,
    DELIVERY_GOAL_A_HEADING_TOLERANCE,
    DELIVERY_GOAL_A_MARKER_FALLBACK,
    DELIVERY_GOAL_B_HEADING_TOLERANCE,
    DELIVERY_GOAL_B_MARKER_FALLBACK,
    DELIVERY_GOAL_DISTANCE_CORRECTION_ATTEMPTS,
    DELIVERY_GOAL_DISTANCE_CORRECTION_MARGIN,
    DELIVERY_GOAL_DISTANCE_REVERSE_ENABLED,
    DELIVERY_GOAL_DISTANCE_REVERSE_HEADING_TOLERANCE,
    DELIVERY_GOAL_DISTANCE_REVERSE_MAX_DISTANCE,
    DELIVERY_GOAL_DISTANCE_REVERSE_MAX_SECONDS,
    DELIVERY_GOAL_DISTANCE_REVERSE_MIN_SECONDS,
    DELIVERY_GOAL_DISTANCE_REVERSE_SECONDS_PER_MAP_UNIT,
    DELIVERY_GOAL_DISTANCE_REVERSE_SPEED,
    DELIVERY_GOAL_PREFERENCE,
    DELIVERY_HEADING_TOLERANCE,
    DELIVERY_MIN_CENTER_GOAL_DISTANCE,
    DELIVERY_MIN_CLAW_GOAL_DISTANCE,
    DELIVERY_PREFER_VISION_GOALS,
    DELIVERY_POSITION_TOLERANCE,
    DELIVERY_RED_CROSS_CLEARANCE_MARGIN,
    DELIVERY_RED_CROSS_RETRY_FRAMES,
    DELIVERY_RED_CROSS_LOOKAHEAD_DISTANCE,
    DELIVERY_REQUIRE_CENTER_POSITION,
    DELIVERY_ROBOT_EDGE_MARGIN,
    DELIVERY_USE_FIXED_GOALS,
    DELIVERY_WAYPOINT_STEP_SIZE,
    HELD_CLAW_RECLOSE_DELAY_SECONDS,
    HELD_CLAW_RECLOSE_ENABLED,
    MAP_HEIGHT,
    MAP_WIDTH,
    PICKUP_FINAL_SYNC_DELAY_SECONDS,
    PICKUP_SETTLE_SECONDS,
    RED_CROSS_BACKOFF_MAX_ATTEMPTS,
    RED_CROSS_WAYPOINT_ACCEPTANCE_RADIUS,
    ROBOT_POSE_RETRY_DELAY_SECONDS,
    ROBOT_POSE_RETRY_FRAMES,
)
from vision_detection import set_vision_path_overlay


def planning_matrix_from_scene(scene):
    if scene is None:
        return create_empty_path_matrix()

    path_matrix = scene.get("path_matrix")

    if path_matrix is not None:
        return path_matrix

    return create_empty_path_matrix()


def capture_delivery_scene_frame(camera):
    scene = capture_vision_scene_frame(camera, "delivery")

    if scene is None:
        return None

    vision_scene = scene["vision_scene"]
    robot_pose = robot_pose_from_sources(vision_scene)
    goals = choose_delivery_goal_markers(vision_scene)
    grappler_point = None

    if vision_scene is not None:
        grappler_point = vision_scene.grappler_point()

    return {
        "path_matrix": scene["path_matrix"],
        "vision_scene": vision_scene,
        "robot_pose": robot_pose,
        "grappler_point": grappler_point,
        "goals": goals,
    }


def capture_delivery_scene(camera, retry_frames=ROBOT_POSE_RETRY_FRAMES):
    return capture_scene_with_robot_pose_retry(
        lambda: capture_delivery_scene_frame(camera),
        "Delivery",
        retry_frames=retry_frames,
    )


def open_claw_detection_in_scene(scene):
    if scene is None:
        return None

    vision_scene = scene.get("vision_scene")

    if vision_scene is None:
        return None

    return vision_scene.open_claw_detection()


def ensure_held_claw_closed(sock, camera, scene, label="Delivery"):
    if not HELD_CLAW_RECLOSE_ENABLED:
        return scene

    detection = open_claw_detection_in_scene(scene)

    if detection is None:
        return scene

    print(
        "{}: vision sees open_claw at {} ({:.2f}) while claw should be closed; sending CLAW_CLOSE".format(
            label,
            detection.point,
            detection.confidence,
        )
    )

    if not send_command(sock, build_claw_close()):
        return None

    time.sleep(HELD_CLAW_RECLOSE_DELAY_SECONDS)

    refreshed_scene = capture_delivery_scene(camera)

    if refreshed_scene is None:
        print("{}: could not refresh camera after re-closing claw; continuing with previous frame".format(label))
        return scene

    refreshed_detection = open_claw_detection_in_scene(refreshed_scene)

    if refreshed_detection is not None:
        print(
            "{}: open_claw is still visible after close command at {} ({:.2f})".format(
                label,
                refreshed_detection.point,
                refreshed_detection.confidence,
            )
        )

    return refreshed_scene


def fixed_delivery_goal_markers():
    return DELIVERY_GOAL_A_MARKER_FALLBACK, DELIVERY_GOAL_B_MARKER_FALLBACK


def choose_delivery_goal_markers(vision_scene=None):
    if DELIVERY_USE_FIXED_GOALS and not DELIVERY_PREFER_VISION_GOALS:
        goal_a, goal_b = fixed_delivery_goal_markers()
        print(
            "Delivery marker detection: using fixed map openings Goal_A={}, Goal_B={}".format(
                goal_a,
                goal_b,
            )
        )
        return goal_a, goal_b

    if vision_scene is not None:
        vision_goals = vision_scene.goal_markers()

        if vision_goals is not None:
            goal_a, goal_b = vision_goals
            print(
                "Delivery marker detection: using vision openings Goal_A={}, Goal_B={}".format(
                    goal_a,
                    goal_b,
                )
            )
            return goal_a, goal_b

    if DELIVERY_USE_FIXED_GOALS:
        goal_a, goal_b = fixed_delivery_goal_markers()
        print(
            "Delivery marker detection: vision did not find both goals; using fixed map openings Goal_A={}, Goal_B={}".format(
                goal_a,
                goal_b,
            )
        )
        return goal_a, goal_b

    print("Delivery marker detection: vision did not find both goals")
    return None


def delivery_goal_heading(goal_name):
    if goal_name == "A":
        return 0.0
    if goal_name == "B":
        return 180.0
    raise ValueError("goal_name must be A or B")


def delivery_goal_heading_tolerance(goal_name):
    if goal_name == "A":
        return float(DELIVERY_GOAL_A_HEADING_TOLERANCE)
    if goal_name == "B":
        return float(DELIVERY_GOAL_B_HEADING_TOLERANCE)
    raise ValueError("goal_name must be A or B")


def delivery_center_is_safe(point, margin=DELIVERY_ROBOT_EDGE_MARGIN):
    row, col = point
    return (
        margin <= row <= MAP_HEIGHT - 1 - margin
        and margin <= col <= MAP_WIDTH - 1 - margin
    )


def delivery_claw_is_clear_of_edge(point, margin=DELIVERY_CLAW_EDGE_MARGIN):
    if point is None:
        return False

    row, col = point
    return (
        margin <= row <= MAP_HEIGHT - 1 - margin
        and margin <= col <= MAP_WIDTH - 1 - margin
    )


def delivery_goal_marker_is_approachable(goal_marker, margin=DELIVERY_ROBOT_EDGE_MARGIN):
    marker_row, _marker_col = goal_marker
    return margin <= marker_row <= MAP_HEIGHT - 1 - margin


def clamp_delivery_center(point):
    clamped_point = clamp_map_point(point, margin=DELIVERY_ROBOT_EDGE_MARGIN)

    if clamped_point != (int(round(point[0])), int(round(point[1]))):
        print(
            "Delivery safety: clamped center target from {} to {}".format(
                (int(round(point[0])), int(round(point[1]))),
                clamped_point,
            )
        )

    return clamped_point


def delivery_claw_target(goal_marker, goal_name, claw_to_marker_distance=DELIVERY_CLAW_TO_MARKER_DISTANCE):
    marker_row, marker_col = goal_marker
    heading = delivery_goal_heading(goal_name)
    heading_rad = math.radians(heading)

    target_col = float(marker_col) - float(claw_to_marker_distance) * math.cos(heading_rad)
    target_row = float(marker_row) - float(claw_to_marker_distance) * math.sin(heading_rad)

    return clamp_map_point((target_row, target_col), margin=5)


def delivery_center_target(
    goal_marker,
    goal_name,
    center_to_marker_distance=DELIVERY_CENTER_TO_MARKER_DISTANCE,
):
    marker_row, marker_col = goal_marker
    heading = delivery_goal_heading(goal_name)
    heading_rad = math.radians(heading)
    center_to_claw_distance = DELIVERY_CENTER_TO_CLAW_DISTANCE

    if center_to_marker_distance != DELIVERY_CENTER_TO_MARKER_DISTANCE:
        center_to_claw_distance = max(
            0.0,
            float(center_to_marker_distance) - float(DELIVERY_CLAW_TO_MARKER_DISTANCE),
        )

    claw_row, claw_col = delivery_claw_target(goal_marker, goal_name)
    target_col = float(claw_col) - float(center_to_claw_distance) * math.cos(heading_rad)
    target_row = float(claw_row) - float(center_to_claw_distance) * math.sin(heading_rad)

    return clamp_delivery_center((target_row, target_col))


def delivery_center_target_from_claw_error(center, claw_point, claw_target):
    center_row, center_col = center
    claw_row, claw_col = claw_point
    target_row, target_col = claw_target

    corrected_center = (
        float(center_row) + float(target_row - claw_row),
        float(center_col) + float(target_col - claw_col),
    )

    return clamp_delivery_center(corrected_center)


def delivery_claw_points_toward_edge(claw_point, heading, margin=DELIVERY_CLAW_EDGE_MARGIN):
    if claw_point is None:
        return False

    row, col = claw_point
    heading_rad = math.radians(float(heading))
    forward_col = math.cos(heading_rad)
    forward_row = math.sin(heading_rad)

    return (
        (forward_col < -0.5 and col <= margin)
        or (forward_col > 0.5 and col >= MAP_WIDTH - 1 - margin)
        or (forward_row < -0.5 and row <= margin)
        or (forward_row > 0.5 and row >= MAP_HEIGHT - 1 - margin)
    )


def delivery_claw_is_on_goal_side(goal_name, marker, claw_point):
    marker_row, marker_col = marker
    claw_row, claw_col = claw_point
    heading_rad = math.radians(delivery_goal_heading(goal_name))
    forward_row = math.sin(heading_rad)
    forward_col = math.cos(heading_rad)
    marker_to_claw_backward = (
        (float(marker_row) - float(claw_row)) * forward_row
        + (float(marker_col) - float(claw_col)) * forward_col
    )

    return marker_to_claw_backward >= -float(DELIVERY_CLAW_POSITION_TOLERANCE)


def delivery_distance_to_goal_wall(goal_name, point):
    _row, col = point

    if goal_name == "A":
        return float(MAP_WIDTH - 1) - float(col)

    if goal_name == "B":
        return float(col)

    raise ValueError("goal_name must be A or B")


def delivery_goal_distance_status(goal_name, center, claw_point):
    center_distance = delivery_distance_to_goal_wall(goal_name, center)
    claw_distance = delivery_distance_to_goal_wall(goal_name, claw_point)
    center_ok = center_distance >= float(DELIVERY_MIN_CENTER_GOAL_DISTANCE)
    claw_ok = claw_distance >= float(DELIVERY_MIN_CLAW_GOAL_DISTANCE)

    return {
        "center_distance": center_distance,
        "claw_distance": claw_distance,
        "center_ok": center_ok,
        "claw_ok": claw_ok,
        "ok": center_ok and claw_ok,
    }


def delivery_safe_center_for_goal_distance(goal_name, center, claw_point):
    center_row, center_col = center
    target_col = float(center_col)
    extra_margin = max(0.0, float(DELIVERY_GOAL_DISTANCE_CORRECTION_MARGIN))
    min_center_distance = float(DELIVERY_MIN_CENTER_GOAL_DISTANCE) + extra_margin
    min_claw_distance = float(DELIVERY_MIN_CLAW_GOAL_DISTANCE) + extra_margin
    claw_distance = delivery_distance_to_goal_wall(goal_name, claw_point)

    if goal_name == "A":
        target_col = min(
            target_col,
            float(MAP_WIDTH - 1) - min_center_distance,
        )

        if claw_distance < min_claw_distance:
            target_col = min(
                target_col,
                float(center_col) - (min_claw_distance - claw_distance),
            )
    elif goal_name == "B":
        target_col = max(
            target_col,
            min_center_distance,
        )

        if claw_distance < min_claw_distance:
            target_col = max(
                target_col,
                float(center_col) + (min_claw_distance - claw_distance),
            )
    else:
        raise ValueError("goal_name must be A or B")

    return clamp_delivery_center((center_row, target_col))


def delivery_goal_option(goal_name, goal_marker):
    if not delivery_goal_marker_is_approachable(goal_marker):
        print(
            "Delivery safety: Goal_{} marker {} is too close to top/bottom wall; skipping".format(
                goal_name,
                goal_marker,
            )
        )
        return None

    claw_target = delivery_claw_target(goal_marker, goal_name)
    waypoint = delivery_center_target(goal_marker, goal_name)

    if not delivery_center_is_safe(waypoint):
        print(
            "Delivery safety: Goal_{} waypoint {} is not safe; skipping".format(
                goal_name,
                waypoint,
            )
        )
        return None

    return {
        "name": goal_name,
        "marker": goal_marker,
        "claw_target": claw_target,
        "waypoint": waypoint,
    }


def choose_delivery_goal(robot_pose, goal_a, goal_b, preference=DELIVERY_GOAL_PREFERENCE):
    options = [
        option for option in (
            delivery_goal_option("A", goal_a),
            delivery_goal_option("B", goal_b),
        )
        if option is not None
    ]

    if preference in ("A", "B"):
        for option in options:
            if option["name"] == preference:
                return option

        print("Delivery: preferred Goal_{} is not safely approachable".format(preference))
        return None

    if robot_pose is None:
        if not options:
            return None

        return options[0]

    robot_center = robot_center_point(robot_pose)

    if not options:
        return None

    return min(options, key=lambda option: point_distance(robot_center, option["waypoint"]))


def delivery_goal_preference_for_ball(_ball_color):
    return DELIVERY_GOAL_PREFERENCE


def turn_delivery_to_heading(sock, camera, target_heading, tolerance_degrees=DELIVERY_HEADING_TOLERANCE):
    for attempt in range(1, 3):
        synced_pose = sync_robot_pose_from_camera(sock, camera)

        if synced_pose is None:
            return False

        _x, _y, current_heading = synced_pose
        turn_angle = normalize_turn_angle(float(target_heading) - float(current_heading))

        print(
            "Delivery heading attempt {}: current={:.1f}, target={:.1f}, turn={:.1f}".format(
                attempt,
                current_heading,
                target_heading,
                turn_angle,
            )
        )

        if abs(turn_angle) <= float(tolerance_degrees):
            return True

        if not send_command(sock, build_turn(int(round(turn_angle)), 0)):
            return False

        time.sleep(PICKUP_FINAL_SYNC_DELAY_SECONDS)

    synced_pose = sync_robot_pose_from_camera(sock, camera)

    if synced_pose is None:
        return False

    _x, _y, current_heading = synced_pose
    final_error = normalize_turn_angle(float(target_heading) - float(current_heading))
    print("Delivery heading final error={:.1f}".format(final_error))

    return abs(final_error) <= float(tolerance_degrees)


def _clamp_float(value, min_value, max_value):
    return max(float(min_value), min(float(max_value), float(value)))


def try_reverse_goal_distance_correction(
    sock,
    camera,
    robot_pose,
    safe_center,
    heading_error,
    label="Delivery goal-distance correction",
):
    if not DELIVERY_GOAL_DISTANCE_REVERSE_ENABLED or robot_pose is None:
        return None

    heading_error = float(heading_error)

    if abs(heading_error) > float(DELIVERY_GOAL_DISTANCE_REVERSE_HEADING_TOLERANCE):
        print(
            "{}: heading error {:.1f} is too large for straight reverse; using normal GOTO".format(
                label,
                heading_error,
            )
        )
        return None

    current_center = robot_center_point(robot_pose)
    reverse_distance = point_distance(current_center, safe_center)

    if reverse_distance <= 2.0:
        print("{}: reverse target is already reached; using normal verification".format(label))
        return None

    if reverse_distance > float(DELIVERY_GOAL_DISTANCE_REVERSE_MAX_DISTANCE):
        print(
            "{}: reverse distance {:.1f} is too large; using normal GOTO".format(
                label,
                reverse_distance,
            )
        )
        return None

    speed = -abs(int(round(DELIVERY_GOAL_DISTANCE_REVERSE_SPEED)))

    if speed == 0:
        print("{}: reverse speed is zero; using normal GOTO".format(label))
        return None

    duration = _clamp_float(
        reverse_distance * float(DELIVERY_GOAL_DISTANCE_REVERSE_SECONDS_PER_MAP_UNIT),
        DELIVERY_GOAL_DISTANCE_REVERSE_MIN_SECONDS,
        DELIVERY_GOAL_DISTANCE_REVERSE_MAX_SECONDS,
    )

    print(
        "{}: reversing instead of GOTO; center={} target={}, distance={:.1f}, "
        "speed={}, seconds={:.2f}".format(
            label,
            current_center,
            safe_center,
            reverse_distance,
            speed,
            duration,
        )
    )

    if not sync_robot_pose_value(sock, robot_pose, label="{} pre-reverse".format(label)):
        return False

    if not send_command(sock, build_setspeed(speed, speed)):
        return False

    time.sleep(duration)

    if not send_command(sock, build_setspeed(0, 0)):
        return False

    time.sleep(PICKUP_FINAL_SYNC_DELAY_SECONDS)
    return sync_robot_pose_from_camera(sock, camera) is not None


def delivery_path_matrix(scene, start_point, target_point):
    if scene is None:
        return None, 0

    base_matrix = planning_matrix_from_scene(scene)
    path_matrix = clone_path_matrix(base_matrix)
    red_cross_count = mark_red_cross_obstacles(
        path_matrix,
        scene.get("vision_scene"),
        margin=DELIVERY_RED_CROSS_CLEARANCE_MARGIN,
    )

    if red_cross_count:
        clear_path_endpoint(path_matrix, start_point, radius=10, value=".")
        clear_path_endpoint(path_matrix, target_point, radius=10, value=".")

    return path_matrix, red_cross_count


def delivery_red_cross_regions(scene):
    base_matrix = planning_matrix_from_scene(scene)
    vision_scene = scene.get("vision_scene") if scene is not None else None
    return red_cross_obstacle_regions(
        base_matrix,
        vision_scene,
        margin=DELIVERY_RED_CROSS_CLEARANCE_MARGIN,
    )


def capture_delivery_scene_with_red_cross(
    camera,
    label="Delivery",
    retry_frames=DELIVERY_RED_CROSS_RETRY_FRAMES,
):
    attempts = max(1, int(retry_frames))
    last_scene = None

    for attempt in range(1, attempts + 1):
        scene = capture_delivery_scene(camera)
        last_scene = scene

        if scene is not None and delivery_red_cross_regions(scene):
            if attempt > 1:
                print("{}: red cross recovered on frame {}".format(label, attempt))
            return scene

        if attempt < attempts:
            print(
                "{}: red cross missing; waiting for next frame ({}/{})".format(
                    label,
                    attempt,
                    attempts,
                )
            )
            time.sleep(ROBOT_POSE_RETRY_DELAY_SECONDS)

    print("{}: red cross still missing after {} frames; refusing blind delivery move".format(label, attempts))
    return last_scene if last_scene is not None and delivery_red_cross_regions(last_scene) else None


def _farthest_path_point_within_distance(robot_path, start_point, max_distance):
    chosen = None

    for point in robot_path[1:]:
        if point_distance(start_point, point) > max_distance:
            break
        chosen = point

    return chosen


def choose_delivery_escape_waypoint(robot_path, start_point, regions):
    if not point_in_obstacle_regions(start_point, regions):
        return None

    previous_point = None

    for point in robot_path[1:]:
        previous_point = point

        if not point_in_obstacle_regions(point, regions):
            print(
                "Delivery dynamic replan: robot starts inside red-cross clearance; escaping via {}".format(
                    point,
                )
            )
            return point

    if previous_point is not None and point_distance(start_point, previous_point) > 2.0:
        print(
            "Delivery dynamic replan: path stays inside clearance for now; continuing escape via {}".format(
                previous_point,
            )
        )
        return previous_point

    return None


def choose_safe_delivery_waypoint(robot_path, start_point, target_point, regions):
    if not path_is_valid(robot_path):
        return None

    escape_point = choose_delivery_escape_waypoint(robot_path, start_point, regions)

    if escape_point is not None:
        return escape_point

    lookahead_point = choose_safe_path_lookahead(
        robot_path,
        start_point,
        regions,
        min_distance=max(
            float(DELIVERY_WAYPOINT_STEP_SIZE),
            float(RED_CROSS_WAYPOINT_ACCEPTANCE_RADIUS),
        ),
        max_distance=float(DELIVERY_RED_CROSS_LOOKAHEAD_DISTANCE),
        acceptance_radius=RED_CROSS_WAYPOINT_ACCEPTANCE_RADIUS,
    )

    if lookahead_point is not None:
        print(
            "Delivery dynamic replan: using red-cross lookahead waypoint {}".format(
                lookahead_point,
            )
        )
        return lookahead_point

    waypoints = simplify_path_for_robot(
        robot_path,
        min_spacing=DELIVERY_WAYPOINT_STEP_SIZE,
    )

    candidates = waypoints[1:] if len(waypoints) > 1 else []
    safe_simplified_point = None

    for point in candidates:
        if segment_intersects_regions(start_point, point, regions):
            print(
                "Delivery dynamic replan: simplified waypoint {} crosses current red cross; trying a closer point".format(
                    point,
                )
            )
            continue
        safe_simplified_point = point

    if safe_simplified_point is not None:
        print(
            "Delivery dynamic replan: using farthest safe simplified waypoint {}".format(
                safe_simplified_point,
            )
        )
        return safe_simplified_point

    safe_point = None

    for point in robot_path[1:]:
        if segment_intersects_regions(start_point, point, regions):
            break
        safe_point = point

    if safe_point is not None and point_distance(start_point, safe_point) > 2.0:
        return safe_point

    raw_step = _farthest_path_point_within_distance(
        robot_path,
        start_point,
        max(8.0, float(DELIVERY_WAYPOINT_STEP_SIZE) / 2.0),
    )

    if raw_step is not None and point_distance(start_point, raw_step) > 2.0:
        print(
            "Delivery dynamic replan: using short raw A* waypoint {} after shortcut checks failed".format(
                raw_step,
            )
        )
        return raw_step

    for point in robot_path[1:]:
        if point_distance(start_point, point) > 2.0:
            print(
                "Delivery dynamic replan: keeping progress with nearest raw A* waypoint {}".format(
                    point,
                )
            )
            return point

    if not segment_intersects_regions(start_point, target_point, regions):
        return target_point

    if len(robot_path) > 1:
        print(
            "Delivery dynamic replan: using final raw A* fallback waypoint {}".format(
                robot_path[1],
            )
        )
        return robot_path[1]

    return target_point


def goto_delivery_target(sock, camera, scene, robot_pose, target_point, label="Delivery waypoint"):
    if robot_pose is None:
        scene = capture_delivery_scene_with_red_cross(camera, label=label)

        if scene is None:
            return False

        robot_pose = scene["robot_pose"]

        if robot_pose is None:
            print("{}: could not detect robot pose for delivery move".format(label))
            reverse_for_missing_robot(sock, label=label)
            return False

    red_cross_backoff_count = 0

    for step in range(1, max(1, int(DELIVERY_DYNAMIC_REPLAN_MAX_STEPS)) + 1):
        fresh_scene = capture_delivery_scene_with_red_cross(
            camera,
            label="{} replan".format(label),
        )

        if fresh_scene is None:
            return False

        robot_pose = fresh_scene["robot_pose"]

        if robot_pose is None:
            print("{}: could not detect robot pose for dynamic replan".format(label))
            reverse_for_missing_robot(sock, label=label)
            return False

        start_point = robot_center_point(robot_pose)

        if point_distance(start_point, target_point) <= 2.0:
            print("{}: already at target {}".format(label, target_point))
            return sync_robot_pose_value(
                sock,
                robot_pose,
                label="{} already-at-target".format(label),
            )

        regions = delivery_red_cross_regions(fresh_scene)

        if not regions:
            print("{}: no red cross in fresh scene; refusing blind delivery move".format(label))
            return False

        crosses_red_cross = segment_intersects_regions(start_point, target_point, regions)

        if not crosses_red_cross:
            print(
                "{} dynamic replan {}: fresh direct segment {} -> {} avoids red cross; using direct GOTO".format(
                    label,
                    step,
                    start_point,
                    target_point,
                )
            )
            set_vision_path_overlay(
                [start_point, target_point],
                label=label,
                color=(0, 255, 255),
            )
            return goto_map_point_with_pose(
                sock,
                camera,
                robot_pose,
                target_point,
                label=label,
            )

        path_matrix, red_cross_count = delivery_path_matrix(fresh_scene, start_point, target_point)

        if red_cross_count <= 0:
            print("{}: red cross disappeared while planning; refusing blind delivery move".format(label))
            return False

        robot_path = A_star(path_matrix, start_point, target_point)

        if not path_is_valid(robot_path):
            print(
                "{}: red cross was detected, but A* could not find a safe route: {}".format(
                    label,
                    robot_path,
                )
            )

            if (
                point_in_obstacle_regions(start_point, regions)
                and red_cross_backoff_count < RED_CROSS_BACKOFF_MAX_ATTEMPTS
                and back_off_from_red_cross(
                    sock,
                    robot_pose=robot_pose,
                    regions=regions,
                    label="{} red-cross escape".format(label),
                )
            ):
                red_cross_backoff_count += 1
                continue

            return False

        next_point = choose_safe_delivery_waypoint(
            robot_path,
            start_point,
            target_point,
            regions,
        )

        if next_point is None:
            print("{}: could not choose a safe next waypoint around the red cross".format(label))

            if (
                point_in_obstacle_regions(start_point, regions)
                and red_cross_backoff_count < RED_CROSS_BACKOFF_MAX_ATTEMPTS
                and back_off_from_red_cross(
                    sock,
                    robot_pose=robot_pose,
                    regions=regions,
                    label="{} red-cross escape".format(label),
                )
            ):
                red_cross_backoff_count += 1
                continue

            return False

        set_vision_path_overlay(
            [
                {
                    "points": robot_path,
                    "label": "{} planned route".format(label),
                    "color": (255, 0, 255),
                },
                {
                    "points": [start_point, next_point],
                    "label": "{} lookahead".format(label),
                    "color": (0, 255, 255),
                },
            ],
            label=label,
        )

        print(
            "{} dynamic replan {}: moving from {} toward {} via {}".format(
                label,
                step,
                start_point,
                target_point,
                next_point,
            )
        )

        waypoint_label = "{} dynamic waypoint {}".format(label, step)

        if not goto_then_sync_with_pre_turn(
            sock,
            camera,
            next_point[0],
            next_point[1],
            label=waypoint_label,
        ):
            return False

    print(
        "{}: dynamic replan did not reach target after {} step(s)".format(
            label,
            int(DELIVERY_DYNAMIC_REPLAN_MAX_STEPS),
        )
    )
    return False


def reverse_delivery_away_from_edge(sock, camera):
    print(
        "Delivery safety: claw is pointed into an edge; reversing for {:.2f}s before turning".format(
            DELIVERY_EDGE_ESCAPE_SECONDS,
        )
    )

    if not sync_robot_pose_from_camera(sock, camera):
        return None

    if not send_command(
        sock,
        build_setspeed(DELIVERY_EDGE_ESCAPE_REVERSE_SPEED, DELIVERY_EDGE_ESCAPE_REVERSE_SPEED),
    ):
        return None

    time.sleep(DELIVERY_EDGE_ESCAPE_SECONDS)

    if not send_command(sock, build_setspeed(0, 0)):
        return None

    time.sleep(PICKUP_FINAL_SYNC_DELAY_SECONDS)
    return sync_robot_pose_from_camera(sock, camera)


def move_to_safe_delivery_staging(sock, camera, scene):
    robot_pose = scene["robot_pose"]
    grappler_point = scene["grappler_point"]
    current_center = robot_center_point(robot_pose)
    safe_center = clamp_delivery_center(current_center)
    needs_staging = not delivery_center_is_safe(current_center) or safe_center != current_center

    if grappler_point is not None:
        _center_x, _center_y, current_heading = robot_pose

        if delivery_claw_points_toward_edge(grappler_point, current_heading):
            robot_pose = reverse_delivery_away_from_edge(sock, camera)

            if robot_pose is None:
                return None

            refreshed_scene = capture_delivery_scene(camera)

            if refreshed_scene is None or refreshed_scene["robot_pose"] is None:
                return robot_pose

            refreshed_scene = ensure_held_claw_closed(
                sock,
                camera,
                refreshed_scene,
                label="Delivery safety",
            )

            if refreshed_scene is None or refreshed_scene["robot_pose"] is None:
                return robot_pose

            scene = refreshed_scene
            robot_pose = scene["robot_pose"]
            grappler_point = scene["grappler_point"]
            current_center = robot_center_point(robot_pose)
            safe_center = clamp_delivery_center(current_center)
            needs_staging = (
                not delivery_center_is_safe(current_center)
                or safe_center != current_center
            )

        if grappler_point is not None and not delivery_claw_is_clear_of_edge(grappler_point):
            safe_grappler = clamp_map_point(grappler_point, margin=DELIVERY_CLAW_EDGE_MARGIN)
            delta_row = safe_grappler[0] - grappler_point[0]
            delta_col = safe_grappler[1] - grappler_point[1]
            safe_center = clamp_delivery_center(
                (
                    float(current_center[0]) + float(delta_row),
                    float(current_center[1]) + float(delta_col),
                )
            )
            needs_staging = True
            print(
                "Delivery safety: claw {} is near an edge; staging center to {}".format(
                    grappler_point,
                    safe_center,
                )
            )

    if not needs_staging:
        return robot_pose

    print(
        "Delivery safety: robot is near an edge at {}; staging inward to {}".format(
            current_center,
            safe_center,
        )
    )

    if not goto_delivery_target(sock, camera, scene, robot_pose, safe_center, label="Delivery safety staging"):
        return None

    return sync_robot_pose_from_camera(sock, camera)


def choose_fresh_delivery_option(sock, camera, goal_name):
    scene = capture_delivery_scene(camera)

    if scene is None:
        print("Delivery verify: could not read camera frame")
        return None, None, None, None

    scene = ensure_held_claw_closed(sock, camera, scene, label="Delivery verify")

    if scene is None:
        print("Delivery verify: could not close claw after open_claw detection")
        return None, None, None, None

    robot_pose = scene["robot_pose"]
    grappler_point = scene["grappler_point"]
    goals = scene["goals"]

    if robot_pose is None:
        print("Delivery verify: could not detect robot pose")
        return scene, None, grappler_point, None

    if goals is None:
        print("Delivery verify: could not detect both goal markers")
        return scene, robot_pose, grappler_point, None

    goal_a, goal_b = goals
    goal_marker = goal_a if goal_name == "A" else goal_b
    return scene, robot_pose, grappler_point, delivery_goal_option(goal_name, goal_marker)


def verify_delivery_alignment(sock, camera, goal_name, initial_option):
    option = initial_option
    max_attempts = max(
        int(DELIVERY_FINAL_CORRECTION_ATTEMPTS),
        int(DELIVERY_GOAL_DISTANCE_CORRECTION_ATTEMPTS),
    )

    for attempt in range(1, max_attempts + 2):
        scene, robot_pose, grappler_point, fresh_option = choose_fresh_delivery_option(sock, camera, goal_name)

        if robot_pose is None:
            if (
                attempt <= max_attempts
                and scene is not None
                and grappler_point is None
                and robot_body_visible(scene["vision_scene"])
            ):
                if not reverse_for_missing_grappler(sock, label="Delivery verify"):
                    return False
                continue
            if attempt <= max_attempts and scene is not None:
                if not reverse_for_missing_robot(sock, label="Delivery verify"):
                    return False
                continue
            return False

        if grappler_point is None:
            print("Delivery verify: could not detect claw; not pushing")
            if attempt <= max_attempts and scene is not None and robot_body_visible(scene["vision_scene"]):
                if not reverse_for_missing_grappler(sock, label="Delivery verify"):
                    return False
                continue
            return False

        if fresh_option is None:
            print("Delivery verify: selected goal is not visible or not safely approachable")
            return False

        option = fresh_option

        center = robot_center_point(robot_pose)
        waypoint = option["waypoint"]
        marker = option["marker"]
        claw_target = option["claw_target"]
        goal_heading = delivery_goal_heading(goal_name)
        heading_tolerance = delivery_goal_heading_tolerance(goal_name)
        _x, _y, current_heading = robot_pose
        position_error = point_distance(center, waypoint)
        claw_error = point_distance(grappler_point, claw_target)
        heading_error = normalize_turn_angle(goal_heading - current_heading)
        claw_on_goal_side = delivery_claw_is_on_goal_side(goal_name, marker, grappler_point)
        position_ok = position_error <= DELIVERY_POSITION_TOLERANCE
        claw_ok = claw_error <= DELIVERY_CLAW_POSITION_TOLERANCE
        heading_ok = abs(heading_error) <= heading_tolerance
        center_position_required = bool(DELIVERY_REQUIRE_CENTER_POSITION)
        goal_distance = delivery_goal_distance_status(goal_name, center, grappler_point)
        goal_distance_ok = goal_distance["ok"]

        print(
            "Delivery verify attempt {}: center={}, claw={}, marker={}, "
            "claw_target={}, waypoint={}, position_error={:.1f}, "
            "claw_error={:.1f}, heading_error={:.1f}, position_ok={}, "
            "claw_ok={}, heading_ok={}, center_required={}, claw_on_goal_side={}, "
            "heading_tolerance={:.1f}, center_goal_distance={:.1f}, "
            "claw_goal_distance={:.1f}, goal_distance_ok={}".format(
                attempt,
                center,
                grappler_point,
                marker,
                claw_target,
                waypoint,
                position_error,
                claw_error,
                heading_error,
                position_ok,
                claw_ok,
                heading_ok,
                center_position_required,
                claw_on_goal_side,
                heading_tolerance,
                goal_distance["center_distance"],
                goal_distance["claw_distance"],
                goal_distance_ok,
            )
        )

        if not delivery_center_is_safe(center):
            print("Delivery safety: center {} is too close to a wall; not pushing".format(center))
            return False

        if not delivery_claw_is_clear_of_edge(grappler_point, margin=5):
            print("Delivery safety: claw {} is touching map edge; not pushing".format(grappler_point))
            return False

        if not goal_distance_ok:
            if attempt > DELIVERY_GOAL_DISTANCE_CORRECTION_ATTEMPTS:
                print(
                    "Delivery safety: still too close to Goal_{}; not pushing".format(
                        goal_name,
                    )
                )
                return False

            safe_center = delivery_safe_center_for_goal_distance(
                goal_name,
                center,
                grappler_point,
            )
            print(
                "Delivery safety: too close to Goal_{} "
                "(center distance {:.1f}, claw distance {:.1f}); moving inward to {}".format(
                    goal_name,
                    goal_distance["center_distance"],
                    goal_distance["claw_distance"],
                    safe_center,
                )
            )

            reverse_result = try_reverse_goal_distance_correction(
                sock,
                camera,
                robot_pose,
                safe_center,
                heading_error,
            )

            if reverse_result is None:
                if not goto_delivery_target(
                    sock,
                    camera,
                    scene,
                    robot_pose,
                    safe_center,
                    label="Delivery goal-distance correction",
                ):
                    return False
            elif not reverse_result:
                return False

            continue

        if claw_ok and heading_ok and claw_on_goal_side and (position_ok or not center_position_required):
            if not position_ok:
                print(
                    "Delivery verify: accepting because claw is aligned; "
                    "center error {:.1f} is only a soft check".format(position_error)
                )
            return True

        if attempt > DELIVERY_FINAL_CORRECTION_ATTEMPTS:
            print("Delivery verify: still not lined up; not pushing")
            return False

        if not claw_ok:
            corrected_waypoint = delivery_center_target_from_claw_error(
                center,
                grappler_point,
                claw_target,
            )
            print(
                "Delivery verify: correcting claw position before push via center {}".format(
                    corrected_waypoint,
                )
            )
            if not goto_map_point_with_pose(
                sock,
                camera,
                robot_pose,
                corrected_waypoint,
                label="Delivery claw correction",
            ):
                return False
        elif center_position_required and not position_ok:
            print("Delivery verify: correcting center position before push")
            if not goto_map_point_with_pose(
                sock,
                camera,
                robot_pose,
                waypoint,
                label="Delivery final correction",
            ):
                return False

        if not heading_ok:
            print("Delivery verify: correcting heading before push")
            if not turn_delivery_to_heading(
                sock,
                camera,
                goal_heading,
                tolerance_degrees=heading_tolerance,
            ):
                return False

    return False


def deliver_held_ball_to_goal(sock, camera, ball_color=None):
    scene = capture_delivery_scene(camera)

    if scene is None:
        print("Delivery: could not read camera frame")
        return False

    scene = ensure_held_claw_closed(sock, camera, scene, label="Delivery start")

    if scene is None:
        print("Delivery: could not close claw after open_claw detection")
        return False

    robot_pose = scene["robot_pose"]
    goals = scene["goals"]

    if robot_pose is None:
        print("Delivery: could not detect robot pose")
        if scene.get("grappler_point") is None and robot_body_visible(scene["vision_scene"]):
            reverse_for_missing_grappler(sock, label="Delivery start")
        else:
            reverse_for_missing_robot(sock, label="Delivery start")
        return False

    if goals is None:
        print("Delivery: could not detect both goal markers")
        return False

    robot_pose = move_to_safe_delivery_staging(sock, camera, scene)

    if robot_pose is None:
        print("Delivery: could not move safely away from edge before scoring")
        return False

    scene = capture_delivery_scene(camera)

    if scene is None:
        print("Delivery: could not refresh camera frame after safety staging")
        return False

    scene = ensure_held_claw_closed(sock, camera, scene, label="Delivery after safety staging")

    if scene is None:
        print("Delivery: could not close claw after safety staging")
        return False

    robot_pose = scene["robot_pose"]
    goals = scene["goals"]

    if robot_pose is None:
        print("Delivery: could not detect robot pose after safety staging")
        if scene.get("grappler_point") is None and robot_body_visible(scene["vision_scene"]):
            reverse_for_missing_grappler(sock, label="Delivery after safety staging")
        else:
            reverse_for_missing_robot(sock, label="Delivery after safety staging")
        return False

    if goals is None:
        print("Delivery: could not detect both goal markers after safety staging")
        return False

    goal_a, goal_b = goals
    goal_preference = delivery_goal_preference_for_ball(ball_color)
    delivery_option = choose_delivery_goal(
        robot_pose,
        goal_a,
        goal_b,
        preference=goal_preference,
    )

    if delivery_option is None:
        print("Delivery: no safely approachable goal")
        return False

    goal_name = delivery_option["name"]
    goal_marker = delivery_option["marker"]
    claw_target = delivery_option["claw_target"]
    delivery_waypoint = delivery_option["waypoint"]
    goal_heading = delivery_goal_heading(goal_name)
    heading_tolerance = delivery_goal_heading_tolerance(goal_name)

    print(
        "Delivery: chosen Goal_{} marker={}, claw_target={}, waypoint={}, heading={:.1f}, tolerance={:.1f}".format(
            goal_name,
            goal_marker,
            claw_target,
            delivery_waypoint,
            goal_heading,
            heading_tolerance,
        )
    )

    if not goto_delivery_target(sock, camera, scene, robot_pose, delivery_waypoint, label="Delivery waypoint"):
        return False

    print("Delivery: aligning to goal heading {:.1f}".format(goal_heading))
    if not turn_delivery_to_heading(
        sock,
        camera,
        goal_heading,
        tolerance_degrees=heading_tolerance,
    ):
        return False

    if not verify_delivery_alignment(sock, camera, goal_name, delivery_option):
        return False

    print("Delivery: stopping before deliver motion")
    if not send_command(sock, build_setspeed(0, 0)):
        return False

    time.sleep(PICKUP_SETTLE_SECONDS)

    print("Delivery: running CLAW_DELIVER")
    return send_command(sock, build_claw_deliver())
