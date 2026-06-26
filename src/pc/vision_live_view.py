#!/usr/bin/env python3
import cv2 as cv

from camera import close_camera, detect_vision_from_warped_frame, open_camera, read_arena_frame
from settings import CAMERA_INDEX, VISION_LIVE_VIEW_ENABLED
from vision_detection import vision_live_view_quit_requested


def main():
    if not VISION_LIVE_VIEW_ENABLED:
        print("Vision live view is disabled. Set GOLFBOT_VISION_LIVE_VIEW=1 to enable it.")

    camera = open_camera(CAMERA_INDEX)

    if camera is None:
        return

    print("Vision live view running. Press q in a camera window to stop.")

    try:
        while camera.isOpened():
            _raw_frame, prepared_frame = read_arena_frame(camera)

            if prepared_frame is None:
                continue

            if prepared_frame is not None:
                detect_vision_from_warped_frame(prepared_frame)
                cv.imshow("Golfbot prepared frame", prepared_frame)

            if vision_live_view_quit_requested() or (cv.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        close_camera(camera)


if __name__ == "__main__":
    main()
