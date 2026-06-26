import os
from datetime import datetime

import cv2 as cv

from settings import (
    VISION_MISSING_DETECTION_DIR,
    VISION_MISSING_DETECTION_SAVE_ENABLED,
)


_saved_missing_detection_since_motion = False
_movement_generation = 0


def mark_robot_moved(reason=None):
    global _saved_missing_detection_since_motion
    global _movement_generation

    _movement_generation += 1
    _saved_missing_detection_since_motion = False

    if reason:
        print("Missing-detection capture: reset after {}".format(reason))


def _safe_name(text):
    cleaned = []

    for character in str(text).lower():
        if character.isalnum():
            cleaned.append(character)
        elif character in ("-", "_"):
            cleaned.append(character)
        else:
            cleaned.append("_")

    return "".join(cleaned).strip("_") or "unknown"


def _missing_reasons(vision_scene, require_claw=True, require_robot_pose=True):
    if vision_scene is None:
        return []

    reasons = []

    if require_claw and vision_scene.grappler_point() is None:
        reasons.append("missing_claw")

    if require_robot_pose and vision_scene.robot_pose() is None:
        reasons.append("missing_robot_pose")

    return reasons


def save_missing_detection_frame(
    frame,
    vision_scene,
    context,
    require_claw=True,
    require_robot_pose=True,
):
    global _saved_missing_detection_since_motion

    if not VISION_MISSING_DETECTION_SAVE_ENABLED:
        return None

    if frame is None:
        return None

    reasons = _missing_reasons(
        vision_scene,
        require_claw=require_claw,
        require_robot_pose=require_robot_pose,
    )

    if not reasons:
        return None

    if _saved_missing_detection_since_motion:
        return None

    os.makedirs(VISION_MISSING_DETECTION_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = "{}_gen{:04d}_{}_{}.png".format(
        timestamp,
        _movement_generation,
        _safe_name(context),
        "_".join(reasons),
    )
    path = os.path.join(VISION_MISSING_DETECTION_DIR, filename)

    if not cv.imwrite(path, frame):
        print("Missing-detection capture: could not save {}".format(path))
        return None

    _saved_missing_detection_since_motion = True
    print(
        "Missing-detection capture: saved {} ({})".format(
            path,
            ", ".join(reasons),
        )
    )
    return path
