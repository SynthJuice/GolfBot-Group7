import time

from camera import (
    detect_vision_from_warped_frame,
    read_arena_frame,
)
from path_obstacles import create_empty_path_matrix
from settings import (
    ROBOT_POSE_RETRY_DELAY_SECONDS,
    ROBOT_POSE_RETRY_FRAMES,
)
from vision_debug_capture import save_missing_detection_frame


def capture_vision_scene_frame(
    camera,
    label,
    require_claw=True,
    require_robot_pose=True,
):
    _raw_frame, warped_frame = read_arena_frame(camera)

    if warped_frame is None:
        return None

    vision_scene = detect_vision_from_warped_frame(warped_frame)
    save_missing_detection_frame(
        warped_frame,
        vision_scene,
        label,
        require_claw=require_claw,
        require_robot_pose=require_robot_pose,
    )

    return {
        "path_matrix": create_empty_path_matrix(),
        "vision_scene": vision_scene,
        "warped_frame": warped_frame,
    }


def robot_pose_from_sources(vision_scene):
    if vision_scene is None:
        return None

    return vision_scene.robot_pose()


def robot_body_visible(vision_scene):
    return vision_scene is not None and vision_scene.best("robot") is not None


def grappler_point_from_sources(vision_scene):
    if vision_scene is None:
        return None

    return vision_scene.grappler_point()


def ball_points_from_sources(vision_scene, ball_color):
    if vision_scene is None:
        return []

    return vision_scene.ball_points(ball_color)


def capture_scene_with_robot_pose_retry(
    capture_frame,
    label,
    retry_frames=ROBOT_POSE_RETRY_FRAMES,
):
    attempts = max(1, int(retry_frames))
    last_scene = None

    for attempt in range(1, attempts + 1):
        scene = capture_frame()
        last_scene = scene

        if scene is None:
            if attempt < attempts:
                print(
                    "{} camera: frame read failed; waiting for next frame ({}/{})".format(
                        label,
                        attempt,
                        attempts,
                    )
                )
                time.sleep(ROBOT_POSE_RETRY_DELAY_SECONDS)
            continue

        if scene["robot_pose"] is not None:
            if attempt > 1:
                print("{} camera: robot pose recovered on frame {}".format(label, attempt))
            return scene

        if attempt < attempts:
            print(
                "{} camera: robot pose missing; waiting for next frame ({}/{})".format(
                    label,
                    attempt,
                    attempts,
                )
            )
            time.sleep(ROBOT_POSE_RETRY_DELAY_SECONDS)

    print("{} camera: robot pose still missing after {} frames".format(label, attempts))
    return last_scene
