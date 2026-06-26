import os

import cv2 as cv
import numpy as np

from settings import (
    CAMERA_WARP_ENABLED,
    CAMERA_INDEX,
    IMAGE_DIR,
    MAP_HEIGHT,
    MAP_WIDTH,
    PERSPECTIVE_SOURCE_POINTS,
    PERSPECTIVE_SOURCE_POINTS_SOURCE,
    WARP_CALIBRATION_PATH,
)


_PERSPECTIVE_TARGET_POINTS = (
    (0, 0),
    (MAP_WIDTH - 1, 0),
    (MAP_WIDTH - 1, MAP_HEIGHT - 1),
    (0, MAP_HEIGHT - 1),
)

warp_matrix = cv.getPerspectiveTransform(
    np.float32(PERSPECTIVE_SOURCE_POINTS),
    np.float32(_PERSPECTIVE_TARGET_POINTS),
)


def ensure_image_dir():
    os.makedirs(IMAGE_DIR, exist_ok=True)


def open_camera(camera_index=CAMERA_INDEX):
    print("using camera.py from : ", __file__)
    print("Trying to open camera : ", camera_index)
    print_warp_calibration_status()

    camera = cv.VideoCapture(camera_index)

    if not camera.isOpened():
        print("Could not open camera index {}".format(camera_index))
        return None

    return camera


def print_warp_calibration_status():
    if not CAMERA_WARP_ENABLED:
        print(
            "Camera warp: disabled by GOLFBOT_USE_WARP; using resized raw frames as {}x{}".format(
                MAP_WIDTH,
                MAP_HEIGHT,
            )
        )
        return

    if PERSPECTIVE_SOURCE_POINTS_SOURCE == "default":
        print(
            "Warp calibration: no {} found; using default points {}".format(
                WARP_CALIBRATION_PATH,
                PERSPECTIVE_SOURCE_POINTS,
            )
        )
    elif PERSPECTIVE_SOURCE_POINTS_SOURCE == "invalid":
        print(
            "Warp calibration: {} is invalid; using default points {}".format(
                WARP_CALIBRATION_PATH,
                PERSPECTIVE_SOURCE_POINTS,
            )
        )
    else:
        print(
            "Warp calibration: loaded {} with points {}".format(
                PERSPECTIVE_SOURCE_POINTS_SOURCE,
                PERSPECTIVE_SOURCE_POINTS,
            )
        )


def close_camera(camera):
    if camera is not None:
        camera.release()
    cv.destroyAllWindows()


def warp_frame(frame):
    if not CAMERA_WARP_ENABLED:
        return cv.resize(frame, (MAP_WIDTH, MAP_HEIGHT), interpolation=cv.INTER_AREA)

    return cv.warpPerspective(frame, warp_matrix, (MAP_WIDTH, MAP_HEIGHT))


def raw_image_point_to_map_point(x, y, frame_shape=None):
    if not CAMERA_WARP_ENABLED:
        if frame_shape is None:
            return (
                int(round(max(0, min(MAP_HEIGHT - 1, y)))),
                int(round(max(0, min(MAP_WIDTH - 1, x)))),
            )

        raw_height, raw_width = frame_shape[:2]
        col_scale = float(MAP_WIDTH - 1) / float(max(1, raw_width - 1))
        row_scale = float(MAP_HEIGHT - 1) / float(max(1, raw_height - 1))
        col = int(round(max(0, min(MAP_WIDTH - 1, float(x) * col_scale))))
        row = int(round(max(0, min(MAP_HEIGHT - 1, float(y) * row_scale))))
        return row, col

    point = np.float32([[[float(x), float(y)]]])
    mapped_point = cv.perspectiveTransform(point, warp_matrix)[0][0]
    col = int(round(max(0, min(MAP_WIDTH - 1, mapped_point[0]))))
    row = int(round(max(0, min(MAP_HEIGHT - 1, mapped_point[1]))))
    return row, col


def read_raw_frame(camera):
    res, frame = camera.read()

    if not res:
        print("Could not read camera frame")
        return None

    return frame


def read_arena_frame(camera):
    raw_frame = read_raw_frame(camera)

    if raw_frame is None:
        return None, None

    return raw_frame, warp_frame(raw_frame)


def read_warped_frame(camera):
    raw_frame = read_raw_frame(camera)

    if raw_frame is None:
        return None

    return warp_frame(raw_frame)


def detect_vision_from_raw_frame(raw_frame):
    from vision_detection import detect_vision_scene

    return detect_vision_scene(
        raw_frame,
        point_mapper=lambda x, y: raw_image_point_to_map_point(x, y, raw_frame.shape),
    )


def detect_vision_from_warped_frame(warped_frame):
    from vision_detection import detect_vision_scene

    return detect_vision_scene(warped_frame)


def save_frame(frame, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv.imwrite(path, frame)
    return path


def show_camera_once(camera):
    warped_frame = read_warped_frame(camera)

    if warped_frame is None:
        return

    cv.imshow("camera", warped_frame)
    cv.waitKey(1)
