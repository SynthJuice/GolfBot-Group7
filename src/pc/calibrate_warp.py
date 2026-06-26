import argparse

import cv2 as cv
import numpy as np

from settings import (
    CAMERA_INDEX,
    MAP_HEIGHT,
    MAP_WIDTH,
    WARP_CALIBRATION_PATH,
)


WINDOW_NAME = "Calibration - click 4 corners"
PREVIEW_WINDOW_NAME = "Warped preview"

clicks = []


def on_mouse(event, x, y, _flags, _param):
    if event == cv.EVENT_LBUTTONDOWN and len(clicks) < 4:
        clicks.append([x, y])
        print("Click {}/4: ({}, {})".format(len(clicks), x, y))


def draw_calibration_overlay(frame):
    display = frame.copy()

    for index, (corner_x, corner_y) in enumerate(clicks):
        cv.circle(display, (corner_x, corner_y), 8, (0, 255, 0), -1)
        cv.putText(
            display,
            str(index + 1),
            (corner_x + 10, corner_y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    for index in range(len(clicks) - 1):
        cv.line(display, tuple(clicks[index]), tuple(clicks[index + 1]), (0, 255, 0), 2)

    if len(clicks) == 4:
        cv.line(display, tuple(clicks[3]), tuple(clicks[0]), (0, 255, 0), 2)

    remaining = 4 - len(clicks)
    cv.putText(
        display,
        "Click {} more corner{}".format(remaining, "s" if remaining != 1 else ""),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 255),
        2,
    )

    return display


def warp_preview(frame):
    source_points = np.float32(clicks)
    target_points = np.float32([
        [0, 0],
        [MAP_WIDTH - 1, 0],
        [MAP_WIDTH - 1, MAP_HEIGHT - 1],
        [0, MAP_HEIGHT - 1],
    ])
    matrix = cv.getPerspectiveTransform(source_points, target_points)
    return cv.warpPerspective(frame, matrix, (MAP_WIDTH, MAP_HEIGHT))


def save_calibration():
    np.savetxt(WARP_CALIBRATION_PATH, np.array(clicks), fmt="%.2f")
    print("Saved warp calibration to {}".format(WARP_CALIBRATION_PATH))
    print("Points: {}".format(clicks))


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate the camera-to-map perspective warp.")
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help="OpenCV camera index. Defaults to CAMERA_INDEX from settings.py.",
    )
    return parser.parse_args()


def main():
    global clicks

    args = parse_args()
    camera = cv.VideoCapture(args.camera)

    if not camera.isOpened():
        print("ERROR: Cannot open camera {}".format(args.camera))
        return

    print("=" * 60)
    print("Perspective Warp Calibration")
    print("=" * 60)
    print("Click the 4 arena corners:")
    print("  1. top-left")
    print("  2. top-right")
    print("  3. bottom-right")
    print("  4. bottom-left")
    print("Press Enter to save, or Esc/r to redo.")
    print("Output:", WARP_CALIBRATION_PATH)
    print()

    cv.namedWindow(WINDOW_NAME)
    cv.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while True:
            clicks = []
            print("Waiting for 4 corner clicks...")

            while len(clicks) < 4:
                ok, frame = camera.read()

                if not ok:
                    continue

                cv.imshow(WINDOW_NAME, draw_calibration_overlay(frame))

                key = cv.waitKey(1) & 0xFF

                if key == 27:
                    clicks = []
                    break

            if len(clicks) < 4:
                continue

            ok, frame = camera.read()

            if not ok:
                print("Could not read preview frame; redo calibration")
                continue

            preview = warp_preview(frame)
            cv.putText(
                preview,
                "Enter=save  Esc/r=redo",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv.imshow(PREVIEW_WINDOW_NAME, preview)

            while True:
                key = cv.waitKey(0) & 0xFF

                if key in (10, 13):
                    save_calibration()
                    return

                if key in (27, ord("r")):
                    print("Redoing calibration...")
                    cv.destroyWindow(PREVIEW_WINDOW_NAME)
                    break

    finally:
        camera.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
