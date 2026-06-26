import math
import time

from camera import detect_vision_from_warped_frame, read_arena_frame, save_frame
from com_protocol import build_goto, build_possync, build_setspeed, build_turn, send_command
from map_utils import clamp_map_point, heading_from_map_points, point_distance, robot_center_point
from scene_analysis import robot_pose_from_sources
from settings import (
    EV3_MAP_HEIGHT,
    EV3_MAP_WIDTH,
    MAP_HEIGHT,
    MAP_WIDTH,
    MISSING_GRAPPLER_REVERSE_ENABLED,
    MISSING_GRAPPLER_REVERSE_SECONDS,
    MISSING_GRAPPLER_REVERSE_SETTLE_SECONDS,
    MISSING_GRAPPLER_REVERSE_SPEED,
    PATH_PRETURN_HEADING_TOLERANCE,
    RED_CROSS_BACKOFF_ENABLED,
    RED_CROSS_BACKOFF_FORWARD_WHEN_BEHIND,
    RED_CROSS_BACKOFF_SECONDS,
    RED_CROSS_BACKOFF_SETTLE_SECONDS,
    RED_CROSS_BACKOFF_SPEED,
    RED_CROSS_ESCAPE_DISTANCE,
    RED_CROSS_ESCAPE_GOTO_ENABLED,
    RED_CROSS_ESCAPE_SETTLE_SECONDS,
    PICKUP_FINAL_HEADING_TOLERANCE,
    PICKUP_FINAL_SYNC_DELAY_SECONDS,
    ROBOT_POSE_RETRY_DELAY_SECONDS,
    ROBOT_POSE_RETRY_FRAMES,
    SYNC_DELAY_SECONDS,
    SYNC_IMAGE_PATH,
)
from path_obstacles import point_in_obstacle_regions
from vision_debug_capture import save_missing_detection_frame
from vision_detection import has_vision_path_overlay, set_vision_path_overlay


_EV3_COORDINATE_WARNING_PRINTED = False


def _scale_axis(value, source_size, target_size):
    value = float(value)
    source_size = int(source_size)
    target_size = int(target_size)

    if source_size == target_size or source_size <= 1 or target_size <= 1:
        return int(round(value))

    return int(round(value * float(target_size - 1) / float(source_size - 1)))


def _print_ev3_coordinate_warning_once():
    global _EV3_COORDINATE_WARNING_PRINTED

    if _EV3_COORDINATE_WARNING_PRINTED:
        return

    if EV3_MAP_WIDTH != MAP_WIDTH or EV3_MAP_HEIGHT != MAP_HEIGHT:
        print(
            "EV3 coordinate scaling enabled: PC map={}x{}, EV3 map={}x{}".format(
                MAP_WIDTH,
                MAP_HEIGHT,
                EV3_MAP_WIDTH,
                EV3_MAP_HEIGHT,
            )
        )

    _EV3_COORDINATE_WARNING_PRINTED = True


def map_xy_to_ev3_xy(x, y):
    _print_ev3_coordinate_warning_once()
    ev3_x = _scale_axis(x, MAP_WIDTH, EV3_MAP_WIDTH)
    ev3_y = _scale_axis(y, MAP_HEIGHT, EV3_MAP_HEIGHT)
    return ev3_x, ev3_y


def map_point_to_ev3_xy(point):
    row, col = point
    return map_xy_to_ev3_xy(col, row)


def map_pose_to_ev3_pose(robot_pose):
    x, y, heading = robot_pose
    ev3_x, ev3_y = map_xy_to_ev3_xy(x, y)
    return ev3_x, ev3_y, heading


def ev3_xy_is_valid(x, y):
    return 0 <= int(round(x)) < EV3_MAP_WIDTH and 0 <= int(round(y)) < EV3_MAP_HEIGHT


def get_robot_pose_from_camera_frame(camera):
    _raw_frame, warped_frame = read_arena_frame(camera)

    if warped_frame is None:
        return None

    save_frame(warped_frame, SYNC_IMAGE_PATH)
    print("Saved sync image:", SYNC_IMAGE_PATH)

    vision_scene = detect_vision_from_warped_frame(warped_frame)
    save_missing_detection_frame(
        warped_frame,
        vision_scene,
        "sync",
        require_claw=False,
        require_robot_pose=True,
    )

    if vision_scene is None:
        print("Camera sync: vision unavailable")
        return None

    robot_pose = robot_pose_from_sources(vision_scene)

    if robot_pose is not None:
        print("Camera sync vision pose:", robot_pose)
        return robot_pose

    print("Camera sync: vision did not detect a full robot pose")

    return None


def get_robot_pose_from_camera(camera, retry_frames=ROBOT_POSE_RETRY_FRAMES):
    attempts = max(1, int(retry_frames))

    for attempt in range(1, attempts + 1):
        pose = get_robot_pose_from_camera_frame(camera)

        if pose is not None:
            if attempt > 1:
                print("Camera sync: robot pose recovered on frame {}".format(attempt))
            return pose

        if attempt < attempts:
            print(
                "Camera sync: robot pose missing; waiting for next frame ({}/{})".format(
                    attempt,
                    attempts,
                )
            )
            time.sleep(ROBOT_POSE_RETRY_DELAY_SECONDS)

    print("Camera sync: robot pose still missing after {} frames".format(attempts))
    return None


def sync_robot_pose_from_camera(sock, camera):
    pose = get_robot_pose_from_camera(camera)

    if pose is None:
        print("Could not detect robot pose from camera")
        return None

    map_x, map_y, heading = pose
    x, y = map_xy_to_ev3_xy(map_x, map_y)
    heading_tenths = int(round(heading * 10))

    print(
        "Camera sync: map=({}, {}), ev3=({}, {}), heading={:.1f}".format(
            int(round(map_x)),
            int(round(map_y)),
            x,
            y,
            heading,
        )
    )

    if not send_command(sock, build_possync(x, y, heading_tenths)):
        return None

    return int(round(map_x)), int(round(map_y)), heading


def sync_robot_from_camera(sock, camera):
    return sync_robot_pose_from_camera(sock, camera) is not None


def sync_robot_pose_value(sock, robot_pose, label="Camera pose"):
    if robot_pose is None:
        return False

    map_x, map_y, heading = robot_pose
    x, y = map_xy_to_ev3_xy(map_x, map_y)
    heading_tenths = int(round(float(heading) * 10))

    print(
        "{} sync: map=({}, {}), ev3=({}, {}), heading={:.1f}".format(
            label,
            int(round(map_x)),
            int(round(map_y)),
            x,
            y,
            heading,
        )
    )
    return send_command(sock, build_possync(x, y, heading_tenths))


def normalize_turn_angle(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


def reverse_for_missing_detection(sock, label="Missing detection", missing_part="vision target"):
    if not MISSING_GRAPPLER_REVERSE_ENABLED:
        print("{}: missing {} reverse is disabled".format(label, missing_part))
        return False

    speed = -abs(int(round(MISSING_GRAPPLER_REVERSE_SPEED)))
    reverse_seconds = max(0.0, float(MISSING_GRAPPLER_REVERSE_SECONDS))
    settle_seconds = max(0.0, float(MISSING_GRAPPLER_REVERSE_SETTLE_SECONDS))

    if reverse_seconds <= 0.0 or speed == 0:
        print("{}: missing {} reverse has no movement configured".format(label, missing_part))
        return False

    print(
        "{}: {} is not visible; reversing at speed {} for {:.2f}s".format(
            label,
            missing_part,
            speed,
            reverse_seconds,
        )
    )

    if not send_command(sock, build_setspeed(speed, speed)):
        return False

    time.sleep(reverse_seconds)

    if not send_command(sock, build_setspeed(0, 0)):
        return False

    if settle_seconds > 0.0:
        time.sleep(settle_seconds)

    return True


def reverse_for_missing_grappler(sock, label="Missing grappler"):
    return reverse_for_missing_detection(
        sock,
        label=label,
        missing_part="grappler/claw",
    )


def reverse_for_missing_robot(sock, label="Missing robot"):
    return reverse_for_missing_detection(
        sock,
        label=label,
        missing_part="robot",
    )


def _red_cross_region_center(region):
    top, left, bottom, right = region
    return ((float(top) + float(bottom)) / 2.0, (float(left) + float(right)) / 2.0)


def _nearest_red_cross_center(point, regions):
    centers = [_red_cross_region_center(region) for region in regions]

    if not centers:
        return None

    return min(centers, key=lambda center: point_distance(point, center))


def _rotate_map_vector(row_delta, col_delta, degrees):
    angle = math.radians(float(degrees))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated_col = float(col_delta) * cos_a - float(row_delta) * sin_a
    rotated_row = float(col_delta) * sin_a + float(row_delta) * cos_a
    return rotated_row, rotated_col


def _unit_away_from_cross(center_point, cross_center, robot_heading):
    row_delta = float(center_point[0]) - float(cross_center[0])
    col_delta = float(center_point[1]) - float(cross_center[1])
    distance = math.hypot(row_delta, col_delta)

    if distance > 0.001:
        return row_delta / distance, col_delta / distance

    fallback_heading = math.radians(float(robot_heading) + 180.0)
    return math.sin(fallback_heading), math.cos(fallback_heading)


def red_cross_escape_candidates(robot_pose, regions):
    if robot_pose is None or not regions:
        return []

    center_point = robot_center_point(robot_pose)
    cross_center = _nearest_red_cross_center(center_point, regions)

    if cross_center is None:
        return []

    _center_x, _center_y, heading = robot_pose
    unit_row, unit_col = _unit_away_from_cross(center_point, cross_center, heading)
    base_distance = max(20.0, float(RED_CROSS_ESCAPE_DISTANCE))
    current_cross_distance = point_distance(center_point, cross_center)
    candidates = []
    seen = set()

    for distance_scale in (1.0, 1.35):
        for angle in (0.0, 35.0, -35.0, 70.0, -70.0, 110.0, -110.0, 180.0):
            row_vector, col_vector = _rotate_map_vector(unit_row, unit_col, angle)
            raw_point = (
                float(center_point[0]) + row_vector * base_distance * distance_scale,
                float(center_point[1]) + col_vector * base_distance * distance_scale,
            )
            point = clamp_map_point(raw_point, margin=15)

            if point in seen or point == center_point:
                continue

            seen.add(point)

            if point_in_obstacle_regions(point, regions):
                continue

            if point_distance(point, cross_center) <= current_cross_distance + 10.0:
                continue

            candidates.append(point)

    return candidates


def goto_red_cross_escape_point(sock, robot_pose, regions, label):
    if not RED_CROSS_ESCAPE_GOTO_ENABLED:
        return False

    candidates = red_cross_escape_candidates(robot_pose, regions)

    if not candidates:
        return False

    center_point = robot_center_point(robot_pose)
    target_point = candidates[0]
    target_x, target_y = map_point_to_ev3_xy(target_point)

    print(
        "{}: escaping red cross with map-space GOTO from {} to {} "
        "(ev3=({}, {}))".format(
            label,
            center_point,
            target_point,
            target_x,
            target_y,
        )
    )
    set_vision_path_overlay(
        [center_point, target_point],
        label="{} escape".format(label),
        color=(0, 255, 255),
    )

    if not sync_robot_pose_value(sock, robot_pose, label="{} escape".format(label)):
        return False

    if not send_command(sock, build_goto(target_x, target_y)):
        return False

    time.sleep(max(0.0, float(RED_CROSS_ESCAPE_SETTLE_SECONDS)))
    return True


def back_off_from_red_cross(sock, robot_pose=None, regions=None, label="Red cross backoff"):
    if not RED_CROSS_BACKOFF_ENABLED:
        print("{}: red cross backoff is disabled".format(label))
        return False

    if goto_red_cross_escape_point(sock, robot_pose, regions, label):
        return True

    speed_magnitude = abs(int(round(RED_CROSS_BACKOFF_SPEED)))
    backoff_seconds = max(0.0, float(RED_CROSS_BACKOFF_SECONDS))
    settle_seconds = max(0.0, float(RED_CROSS_BACKOFF_SETTLE_SECONDS))

    if backoff_seconds <= 0.0 or speed_magnitude == 0:
        print("{}: red cross backoff has no movement configured".format(label))
        return False

    direction = -1
    relation = "reversing (cross assumed ahead)"

    if robot_pose is not None and regions:
        center_point = robot_center_point(robot_pose)
        cross_center = _nearest_red_cross_center(center_point, regions)

        if cross_center is not None:
            _center_x, _center_y, heading = robot_pose
            bearing_to_cross = heading_from_map_points(center_point, cross_center)
            angle_to_cross = normalize_turn_angle(float(bearing_to_cross) - float(heading))

            if abs(angle_to_cross) > 90.0 and RED_CROSS_BACKOFF_FORWARD_WHEN_BEHIND:
                direction = 1
                relation = "driving forward (cross behind, angle {:.1f})".format(angle_to_cross)
            else:
                relation = "reversing (cross ahead, angle {:.1f})".format(angle_to_cross)

    speed = direction * speed_magnitude

    print(
        "{}: moving away from the red cross by {} at speed {} for {:.2f}s".format(
            label,
            relation,
            speed,
            backoff_seconds,
        )
    )

    if not send_command(sock, build_setspeed(speed, speed)):
        return False

    time.sleep(backoff_seconds)

    if not send_command(sock, build_setspeed(0, 0)):
        return False

    if settle_seconds > 0.0:
        time.sleep(settle_seconds)

    return True


def turn_robot_to_heading(sock, camera, target_heading, tolerance_degrees=PICKUP_FINAL_HEADING_TOLERANCE):
    synced_pose = sync_robot_pose_from_camera(sock, camera)

    if synced_pose is None:
        return False

    _x, _y, current_heading = synced_pose
    turn_angle = normalize_turn_angle(float(target_heading) - float(current_heading))

    print(
        "Pickup final alignment: current_heading={:.1f}, target_heading={:.1f}, "
        "turn_angle={:.1f}".format(
            current_heading,
            target_heading,
            turn_angle,
        )
    )

    if abs(turn_angle) <= float(tolerance_degrees):
        print("Pickup final alignment: already within tolerance")
        return True

    if not send_command(sock, build_turn(int(round(turn_angle)), 0)):
        return False

    time.sleep(PICKUP_FINAL_SYNC_DELAY_SECONDS)

    return sync_robot_from_camera(sock, camera)


def goto_map_point_with_pose(sock, camera, robot_pose, target_point, label="Delivery GOTO"):
    target_row, target_col = clamp_map_point(target_point, margin=5)
    target_x, target_y = map_point_to_ev3_xy((target_row, target_col))

    print(
        "{}: target center map=({}, {}), ev3=({}, {})".format(
            label,
            target_col,
            target_row,
            target_x,
            target_y,
        )
    )

    if robot_pose is not None:
        if not has_vision_path_overlay():
            set_vision_path_overlay(
                [robot_center_point(robot_pose), (target_row, target_col)],
                label=label,
                color=(0, 255, 255),
            )
        if not sync_robot_pose_value(sock, robot_pose, label="{} pre-GOTO".format(label)):
            return False
    else:
        if not sync_robot_from_camera(sock, camera):
            return False

    if not send_command(sock, build_goto(target_x, target_y)):
        return False

    time.sleep(SYNC_DELAY_SECONDS)
    return sync_robot_from_camera(sock, camera)


def _pre_turn_for_goto(sock, camera, robot_pose, target_point, label, tolerance_degrees):
    start_point = robot_center_point(robot_pose)

    if point_distance(start_point, target_point) <= 2.0:
        print("{} pre-turn: already at target".format(label))
        return robot_pose

    target_heading = heading_from_map_points(start_point, target_point)
    _map_x, _map_y, current_heading = robot_pose
    turn_angle = normalize_turn_angle(float(target_heading) - float(current_heading))
    tolerance_degrees = float(tolerance_degrees)

    print(
        "{} pre-turn: center={}, target={}, current_heading={:.1f}, "
        "target_heading={:.1f}, turn={:.1f}, tolerance={:.1f}".format(
            label,
            start_point,
            target_point,
            current_heading,
            target_heading,
            turn_angle,
            tolerance_degrees,
        )
    )

    if abs(turn_angle) <= tolerance_degrees:
        return robot_pose

    if not send_command(sock, build_turn(int(round(turn_angle)), 0)):
        return None

    time.sleep(PICKUP_FINAL_SYNC_DELAY_SECONDS)
    synced_pose = sync_robot_pose_from_camera(sock, camera)

    if synced_pose is None:
        return None

    _synced_x, _synced_y, synced_heading = synced_pose
    final_error = normalize_turn_angle(float(target_heading) - float(synced_heading))

    print(
        "{} pre-turn: verified_heading={:.1f}, error={:.1f}".format(
            label,
            synced_heading,
            final_error,
        )
    )

    if abs(final_error) > tolerance_degrees:
        print(
            "{} pre-turn: heading is still outside tolerance; refusing forward GOTO".format(
                label,
            )
        )
        return None

    return synced_pose


def goto_map_point_with_pose_pre_turn(
    sock,
    camera,
    robot_pose,
    target_point,
    label="Path GOTO",
    tolerance_degrees=PATH_PRETURN_HEADING_TOLERANCE,
):
    target_row, target_col = clamp_map_point(target_point, margin=5)
    target_point = (target_row, target_col)
    target_x, target_y = map_point_to_ev3_xy(target_point)

    print(
        "{}: target center map=({}, {}), ev3=({}, {})".format(
            label,
            target_col,
            target_row,
            target_x,
            target_y,
        )
    )

    if robot_pose is not None:
        if not sync_robot_pose_value(sock, robot_pose, label="{} pre-GOTO".format(label)):
            return False
        pose = robot_pose
    else:
        pose = sync_robot_pose_from_camera(sock, camera)

        if pose is None:
            return False

    if not has_vision_path_overlay():
        set_vision_path_overlay(
            [robot_center_point(pose), target_point],
            label=label,
            color=(0, 255, 255),
        )

    pose = _pre_turn_for_goto(
        sock,
        camera,
        pose,
        target_point,
        label,
        tolerance_degrees,
    )

    if pose is None:
        return False

    if not send_command(sock, build_goto(target_x, target_y)):
        return False

    time.sleep(SYNC_DELAY_SECONDS)
    return sync_robot_from_camera(sock, camera)


def goto_then_sync_with_pre_turn(
    sock,
    camera,
    row,
    col,
    label="Path GOTO",
    tolerance_degrees=PATH_PRETURN_HEADING_TOLERANCE,
):
    return goto_map_point_with_pose_pre_turn(
        sock,
        camera,
        None,
        (row, col),
        label=label,
        tolerance_degrees=tolerance_degrees,
    )
