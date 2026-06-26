#!/usr/bin/env python3
import os
import socket
import time
from dataclasses import dataclass

import cv2 as cv

from camera import (
    close_camera,
    detect_vision_from_warped_frame,
    ensure_image_dir,
    open_camera,
    save_frame,
    warp_frame,
)
from collection_algorithm import A_star
from com_protocol import HOST, PORT, build_handshake, build_mapsize, send_command
from delivery import deliver_held_ball_to_goal
from map_utils import path_is_valid, point_distance, robot_center_point
from path_obstacles import (
    clear_path_endpoint,
    clear_path_endpoint_preserving_obstacles,
    clone_path_matrix,
    create_empty_path_matrix,
    mark_red_cross_obstacles,
    point_in_obstacle_regions,
    red_cross_obstacle_regions,
)
from pickup import approach_ball_and_close_claw
from robot_sync import back_off_from_red_cross, reverse_for_missing_grappler, reverse_for_missing_robot
from scene_analysis import robot_body_visible
from settings import (
    CAMERA_INDEX,
    EV3_MAP_HEIGHT,
    EV3_MAP_WIDTH,
    FRAME_CAPTURE_INTERVAL_SECONDS,
    IMAGE_DIR,
    MAP_HEIGHT,
    MAP_WIDTH,
    PICKUP_BALL_COLORS,
    PICKUP_BALL_ENDPOINT_CLEAR_RADIUS,
    PICKUP_CORNER_PRIORITY_MARGIN,
    PICKUP_RED_CROSS_CLEARANCE_MARGIN,
    PICKUP_SAFE_BALL_PRIORITY_ENABLED,
    PICKUP_WALL_PRIORITY_MARGIN,
    STARTUP_DELAY_SECONDS,
    STOP_AFTER_SUCCESSFUL_DELIVERY,
)
from vision_debug_capture import save_missing_detection_frame
from vision_detection import (
    clear_vision_path_overlay,
    set_vision_path_overlay,
    vision_live_view_quit_requested,
)


@dataclass
class AutonomousState:
    image_count: int = 0
    path_executed: bool = False
    pickup_started: bool = False
    held_ball_color: str = None
    begin_time: float = 0.0
    last_capture_time: float = 0.0


def startup_delay_has_elapsed(state, now):
    return now - state.begin_time >= STARTUP_DELAY_SECONDS


def capture_interval_has_elapsed(state, now):
    if now - state.last_capture_time < FRAME_CAPTURE_INTERVAL_SECONDS:
        return False

    state.last_capture_time = now
    return True


def save_detection_frame(state, warped_frame):
    image_name = f"{state.image_count}.png"
    full_path = os.path.join(IMAGE_DIR, image_name)
    save_frame(warped_frame, full_path)
    state.image_count += 1

    print("Vi tager et billede")


def capture_detection_scene(state, warped_frame):
    save_detection_frame(state, warped_frame)
    vision_scene = detect_vision_from_warped_frame(warped_frame)
    save_missing_detection_frame(warped_frame, vision_scene, "main")

    return {
        "path_matrix": create_empty_path_matrix(),
        "vision_scene": vision_scene,
    }


def prepare_pickup_path_matrix(path_matrix, grapler_point, ball_point, vision_scene=None):
    if path_matrix is None:
        path_matrix = create_empty_path_matrix()

    path_matrix = clone_path_matrix(path_matrix)
    pickup_cross_regions = red_cross_obstacle_regions(
        path_matrix,
        vision_scene,
        margin=PICKUP_RED_CROSS_CLEARANCE_MARGIN,
    )
    mark_red_cross_obstacles(
        path_matrix,
        vision_scene,
        margin=PICKUP_RED_CROSS_CLEARANCE_MARGIN,
    )
    clear_path_endpoint(path_matrix, grapler_point, radius=8, value=".")
    clear_pickup_ball_endpoint(
        path_matrix,
        path_matrix,
        ball_point,
        radius=PICKUP_BALL_ENDPOINT_CLEAR_RADIUS,
        blocked_regions=pickup_cross_regions,
    )
    return path_matrix


def clear_pickup_ball_endpoint(
    path_matrix,
    original_matrix,
    ball_point,
    radius,
    value="W",
    blocked_regions=None,
):
    clear_path_endpoint_preserving_obstacles(
        path_matrix,
        original_matrix,
        ball_point,
        radius=radius,
        value=value,
        blocked_regions=blocked_regions,
    )


def ball_wall_distances(point):
    row, col = point
    row = float(row)
    col = float(col)
    return {
        "top": row,
        "bottom": float(MAP_HEIGHT - 1) - row,
        "left": col,
        "right": float(MAP_WIDTH - 1) - col,
    }


def ball_safety_priority(point):
    if not PICKUP_SAFE_BALL_PRIORITY_ENABLED:
        return 0, "normal", None

    distances = ball_wall_distances(point)
    wall_distance = min(distances.values())
    corner_margin = max(0.0, float(PICKUP_CORNER_PRIORITY_MARGIN))
    wall_margin = max(0.0, float(PICKUP_WALL_PRIORITY_MARGIN))
    near_top_or_bottom = (
        distances["top"] <= corner_margin
        or distances["bottom"] <= corner_margin
    )
    near_left_or_right = (
        distances["left"] <= corner_margin
        or distances["right"] <= corner_margin
    )

    if corner_margin > 0.0 and near_top_or_bottom and near_left_or_right:
        return 2, "corner", wall_distance

    if wall_margin > 0.0 and wall_distance <= wall_margin:
        return 1, "wall", wall_distance

    return 0, "open", wall_distance


def sort_balls_by_pickup_priority(ball_targets, grapler_point, robot_pose):
    paired_targets = []

    for target in ball_targets:
        point = target["point"]
        distance = point_distance(grapler_point, point)
        safety_priority, safety_label, wall_distance = ball_safety_priority(point)
        paired_targets.append(
            (
                safety_priority,
                distance,
                safety_label,
                wall_distance,
                target,
            )
        )

    paired_targets.sort(key=lambda item: (item[0], item[1]))
    sorted_targets = [
        target
        for (
            _safety_priority,
            _distance,
            _safety_label,
            _wall_distance,
            target,
        ) in paired_targets
    ]

    print(
        "Pickup ball targets by priority:",
        [
            "{}@{}:safety={}({}),distance={:.1f}".format(
                target["color"],
                target["point"],
                safety_label,
                (
                    "wall_distance={:.1f}".format(wall_distance)
                    if wall_distance is not None
                    else "wall_distance=n/a"
                ),
                distance,
            )
            for (
                safety_priority,
                distance,
                safety_label,
                wall_distance,
                target,
            ) in paired_targets
        ],
    )
    return sorted_targets


def detect_ball_points_for_color(vision_scene, ball_color):
    ball_points = []

    if vision_scene is not None:
        ball_points = vision_scene.ball_points(ball_color)

        if ball_points:
            print("Vision {} balls: {}".format(ball_color, ball_points))

    return ball_points


def detect_pickup_ball_targets(vision_scene=None):
    ball_targets = []

    for ball_color in PICKUP_BALL_COLORS:
        for point in detect_ball_points_for_color(vision_scene, ball_color):
            ball_targets.append(
                {
                    "color": ball_color,
                    "point": point,
                }
            )

    return ball_targets


def detect_pickup_target(vision_scene=None, sock=None, path_matrix=None):
    ball_targets = detect_pickup_ball_targets(vision_scene)

    if path_matrix is None:
        path_matrix = create_empty_path_matrix()

    grapler_point = None

    if vision_scene is not None:
        grapler_point = vision_scene.grappler_point()

        if grapler_point is not None:
            print("Vision grapler:", grapler_point)

    print(grapler_point)

    if grapler_point is None:
        print("No grapler detected; cannot collect ball")
        if sock is not None and robot_body_visible(vision_scene):
            reverse_for_missing_grappler(sock, label="Pickup target")
        return None

    current_robot_pose = None

    if vision_scene is not None:
        current_robot_pose = vision_scene.robot_pose()

        if current_robot_pose is not None:
            print("Vision robot pose:", current_robot_pose)

    if current_robot_pose is None:
        print("No robot pose detected; cannot collect ball")
        if sock is not None:
            reverse_for_missing_robot(sock, label="Pickup target")
        return None

    pickup_cross_regions = red_cross_obstacle_regions(
        path_matrix,
        vision_scene,
        margin=PICKUP_RED_CROSS_CLEARANCE_MARGIN,
    )

    if sock is not None and pickup_cross_regions:
        robot_point = robot_center_point(current_robot_pose)
        robot_inside_clearance = point_in_obstacle_regions(robot_point, pickup_cross_regions)
        grappler_inside_clearance = point_in_obstacle_regions(grapler_point, pickup_cross_regions)

        if robot_inside_clearance or grappler_inside_clearance:
            blocked_part = "robot center" if robot_inside_clearance else "claw"
            print(
                "Pickup target: {} is inside the red-cross clearance; escaping before planning".format(
                    blocked_part,
                )
            )
            back_off_from_red_cross(
                sock,
                robot_pose=current_robot_pose,
                regions=pickup_cross_regions,
                label="Pickup target",
            )
            return None

    if not ball_targets:
        print("No pickup balls detected for colors {}".format(PICKUP_BALL_COLORS))
        return None

    ball_targets = sort_balls_by_pickup_priority(
        ball_targets,
        grapler_point,
        current_robot_pose,
    )
    for selected_ball in ball_targets:
        selected_ball_point = selected_ball["point"]
        selected_ball_color = selected_ball["color"]

        if point_in_obstacle_regions(selected_ball_point, pickup_cross_regions):
            print(
                "Skipping pickup target color={}, point={}: too close to red cross for grappler clearance".format(
                    selected_ball_color,
                    selected_ball_point,
                )
            )
            continue

        pickup_matrix = prepare_pickup_path_matrix(
            path_matrix,
            grapler_point,
            selected_ball_point,
            vision_scene,
        )
        robot_path = A_star(pickup_matrix, grapler_point, selected_ball_point)

        if not path_is_valid(robot_path):
            print(
                "Skipping pickup target color={}, point={}: no valid path ({})".format(
                    selected_ball_color,
                    selected_ball_point,
                    robot_path,
                )
            )
            continue

        print(
            "Selected pickup target: color={}, point={}, path_points={}".format(
                selected_ball_color,
                selected_ball_point,
                len(robot_path),
            )
        )
        set_vision_path_overlay(
            [
                {
                    "points": robot_path,
                    "label": "Pickup A* path",
                    "color": (255, 0, 255),
                },
                {
                    "points": [grapler_point, selected_ball_point],
                    "label": "Pickup direct claw line",
                    "color": (0, 255, 255),
                },
            ],
            label="Pickup path",
        )

        return {
            "grapler_point": grapler_point,
            "robot_pose": current_robot_pose,
            "robot_path": robot_path,
            "ball_color": selected_ball_color,
            "ball_point": selected_ball_point,
        }

    print("No reachable pickup balls found; all detected targets were blocked")
    return None


def handle_pickup_and_delivery(sock, camera, state, pickup_target):
    if state.path_executed:
        return False

    pickup_success = approach_ball_and_close_claw(
        sock,
        camera,
        ball_color=pickup_target["ball_color"],
        open_claw=not state.pickup_started,
        target_ball_point=pickup_target["ball_point"],
    )

    state.pickup_started = True

    if not pickup_success:
        print(
            "Pickup attempt did not finish cleanly; will retry from the current position without reopening the claw"
        )
        return False

    print("Pickup succeeded; starting delivery")
    state.path_executed = True
    state.pickup_started = False
    state.held_ball_color = pickup_target["ball_color"]

    return retry_delivery(sock, camera, state)


def retry_delivery(sock, camera, state):
    delivery_success = deliver_held_ball_to_goal(
        sock,
        camera,
        ball_color=state.held_ball_color,
    )

    if delivery_success:
        print("Delivery complete")

        if STOP_AFTER_SUCCESSFUL_DELIVERY:
            return True

        state.path_executed = False
        state.pickup_started = False
        state.held_ball_color = None
        return False

    print("Delivery did not finish; keeping claw closed and retrying on a later frame")
    state.path_executed = True
    state.pickup_started = False
    return False


def print_debug_detections(vision_scene=None):
    if vision_scene is not None:
        print("Vision detections:", vision_scene.summary())


def show_frame_and_should_quit(warped_frame):
    cv.imshow("camera", warped_frame)
    return vision_live_view_quit_requested() or (cv.waitKey(1) & 0xFF) == ord("q")


def run_autonomous_camera():
    start_time = time.time()
    state = AutonomousState(
        begin_time=start_time,
        last_capture_time=start_time,
    )

    ensure_image_dir()

    camera = open_camera(CAMERA_INDEX)
    if camera is None:
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.connect((HOST, PORT))
        if not send_command(sock, build_handshake()):
            return
        if not send_command(sock, build_mapsize(EV3_MAP_HEIGHT, EV3_MAP_WIDTH)):
            return

        camera.read()

        while camera.isOpened():
            res, frame = camera.read()

            if not res:
                continue

            warped_frame = warp_frame(frame)
            now = time.time()

            if startup_delay_has_elapsed(state, now) and capture_interval_has_elapsed(state, now):
                scene = capture_detection_scene(state, warped_frame)
                path_matrix = scene["path_matrix"]
                vision_scene = scene["vision_scene"]

                if state.path_executed:
                    if retry_delivery(sock, camera, state):
                        break
                    continue

                pickup_target = detect_pickup_target(
                    vision_scene,
                    sock=sock,
                    path_matrix=path_matrix,
                )

                if pickup_target is None:
                    clear_vision_path_overlay()
                    continue

                if handle_pickup_and_delivery(sock, camera, state, pickup_target):
                    break

                print_debug_detections(vision_scene)

            if show_frame_and_should_quit(warped_frame):
                break

    finally:
        sock.close()
        close_camera(camera)


def main():
    run_autonomous_camera()


if __name__ == "__main__":
    main()
