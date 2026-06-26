#!/usr/bin/env python3
import argparse
import math
import os
import socket
import time

from camera import close_camera, open_camera
from com_protocol import (
    HOST,
    PORT,
    build_handshake,
    build_mapsize,
    build_motioncal,
    build_raw_drive,
    build_turn,
    send_command,
)
from robot_sync import get_robot_pose_from_camera, normalize_turn_angle, sync_robot_pose_value
from settings import CAMERA_INDEX, EV3_MAP_HEIGHT, EV3_MAP_WIDTH, MAP_HEIGHT, MAP_WIDTH


DEFAULT_TURN_SCALE = 2.35
DEFAULT_DRIVE_SCALE = 5.2

LOCAL_EV3_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ev3"))
LOCAL_LEGACY_TURN_CALIBRATION_PATH = os.path.join(LOCAL_EV3_DIR, "calibration.txt")
LOCAL_MOTION_CALIBRATION_PATH = os.path.join(LOCAL_EV3_DIR, "motion_calibration.txt")

MIN_TURN_MEASUREMENT_DEGREES = 8.0
MIN_DRIVE_MEASUREMENT_UNITS = 8.0
MIN_CORRECTION_FACTOR = 0.4
MAX_CORRECTION_FACTOR = 2.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate EV3 turn and drive scaling using the overhead camera.",
    )
    parser.add_argument("--host", default=HOST, help="EV3 host name or IP address.")
    parser.add_argument("--port", type=int, default=PORT, help="EV3 TCP port.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="OpenCV camera index.")
    parser.add_argument("--iterations", type=int, default=5, help="Calibration passes per phase.")
    parser.add_argument("--turn-angle", type=float, default=90.0, help="Turn angle per pass.")
    parser.add_argument(
        "--drive-distance",
        type=float,
        default=120.0,
        help="Preferred straight-drive distance in map units.",
    )
    parser.add_argument("--turn-speed", type=int, default=20, help="EV3 turn speed percent.")
    parser.add_argument("--drive-speed", type=int, default=30, help="EV3 drive speed percent.")
    parser.add_argument(
        "--settle",
        type=float,
        default=0.45,
        help="Seconds to wait after each move before reading the camera.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=80.0,
        help="Minimum map-unit margin from arena edges during drive calibration.",
    )
    parser.add_argument(
        "--turn-scale",
        type=float,
        default=None,
        help="Starting DEGREES_PER_TURN_DEGREE. Defaults to saved/local value.",
    )
    parser.add_argument(
        "--drive-scale",
        type=float,
        default=None,
        help="Starting DEGREES_PER_MAP_UNIT. Defaults to saved/local value.",
    )
    parser.add_argument("--skip-turn", action="store_true", help="Only calibrate straight driving.")
    parser.add_argument("--skip-drive", action="store_true", help="Only calibrate turning.")
    return parser.parse_args()


def read_float_file(path):
    try:
        with open(path, "r") as value_file:
            return float(value_file.read().strip())
    except (OSError, ValueError):
        return None


def load_local_motion_calibration():
    turn_scale = read_float_file(LOCAL_LEGACY_TURN_CALIBRATION_PATH)

    if turn_scale is None:
        turn_scale = DEFAULT_TURN_SCALE

    drive_scale = DEFAULT_DRIVE_SCALE

    try:
        with open(LOCAL_MOTION_CALIBRATION_PATH, "r") as calibration_file:
            lines = calibration_file.readlines()
    except OSError:
        return turn_scale, drive_scale

    try:
        for line in lines:
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            if "=" in stripped:
                name, value = stripped.split("=", 1)
                name = name.strip().lower()
                value = float(value.strip())

                if name in ("degrees_per_turn_degree", "turn"):
                    turn_scale = value
                elif name in ("degrees_per_map_unit", "drive"):
                    drive_scale = value

            else:
                parts = stripped.split()

                if len(parts) >= 2:
                    turn_scale = float(parts[0])
                    drive_scale = float(parts[1])

    except ValueError:
        print("Local motion calibration is invalid; using defaults/legacy values")
        return turn_scale, DEFAULT_DRIVE_SCALE

    return turn_scale, drive_scale


def save_local_motion_calibration(turn_scale, drive_scale):
    os.makedirs(LOCAL_EV3_DIR, exist_ok=True)

    with open(LOCAL_MOTION_CALIBRATION_PATH, "w") as calibration_file:
        calibration_file.write("degrees_per_turn_degree={:.6f}\n".format(float(turn_scale)))
        calibration_file.write("degrees_per_map_unit={:.6f}\n".format(float(drive_scale)))

    with open(LOCAL_LEGACY_TURN_CALIBRATION_PATH, "w") as calibration_file:
        calibration_file.write("{:.6f}\n".format(float(turn_scale)))

    print("Saved local motion calibration:", LOCAL_MOTION_CALIBRATION_PATH)
    print("Updated local legacy turn calibration:", LOCAL_LEGACY_TURN_CALIBRATION_PATH)


def correction_is_reasonable(correction):
    return MIN_CORRECTION_FACTOR <= correction <= MAX_CORRECTION_FACTOR


def pose_distance(first_pose, second_pose):
    first_x, first_y, _first_heading = first_pose
    second_x, second_y, _second_heading = second_pose
    return math.hypot(float(second_x) - float(first_x), float(second_y) - float(first_y))


def projected_drive_target(pose, distance):
    x, y, heading = pose
    heading_rad = math.radians(float(heading))
    return (
        float(x) + float(distance) * math.cos(heading_rad),
        float(y) + float(distance) * math.sin(heading_rad),
    )


def point_inside_drive_margin(x, y, margin):
    return (
        float(margin) <= float(x) <= float(MAP_WIDTH) - float(margin) and
        float(margin) <= float(y) <= float(MAP_HEIGHT) - float(margin)
    )


def choose_safe_drive_distance(pose, preferred_distance, margin):
    preferred_distance = abs(float(preferred_distance))

    for scale in (1.0, 0.75, 0.5, 0.35, 0.25):
        distance = preferred_distance * scale

        for sign in (1.0, -1.0):
            signed_distance = distance * sign
            target_x, target_y = projected_drive_target(pose, signed_distance)

            if point_inside_drive_margin(target_x, target_y, margin):
                return signed_distance

    return None


def read_pose(camera, label):
    print("{}: reading camera pose".format(label))
    pose = get_robot_pose_from_camera(camera)

    if pose is None:
        print("{}: robot pose unavailable".format(label))
        return None

    x, y, heading = pose
    print("{}: x={:.1f}, y={:.1f}, heading={:.1f}".format(label, x, y, heading))
    return pose


def send_required_command(sock, packet, description):
    if send_command(sock, packet):
        return True

    print("{} failed; stopping calibration".format(description))
    return False


def apply_motion_calibration(sock, turn_scale, drive_scale):
    print("Applying calibration turn={:.6f}, drive={:.6f}".format(turn_scale, drive_scale))
    return send_required_command(
        sock,
        build_motioncal(turn_scale, drive_scale),
        "MOTIONCAL",
    )


def calibrate_turn(sock, camera, turn_scale, drive_scale, args):
    print()
    print("=" * 60)
    print("Turn calibration")
    print("=" * 60)

    iterations = max(1, int(args.iterations))
    requested_magnitude = abs(float(args.turn_angle))

    for index in range(iterations):
        direction = 1.0 if index % 2 == 0 else -1.0
        requested_angle = requested_magnitude * direction

        before_pose = read_pose(camera, "Turn {} before".format(index + 1))

        if before_pose is None:
            break

        if not sync_robot_pose_value(sock, before_pose, "Turn {}".format(index + 1)):
            break

        if not send_required_command(
            sock,
            build_turn(int(round(requested_angle)), int(args.turn_speed)),
            "TURN",
        ):
            break

        time.sleep(float(args.settle))
        after_pose = read_pose(camera, "Turn {} after".format(index + 1))

        if after_pose is None:
            break

        measured_angle = normalize_turn_angle(float(after_pose[2]) - float(before_pose[2]))
        measured_magnitude = abs(measured_angle)

        if measured_magnitude < MIN_TURN_MEASUREMENT_DEGREES:
            print("Turn {} skipped: measured only {:.2f} degrees".format(
                index + 1,
                measured_magnitude,
            ))
            continue

        correction = requested_magnitude / measured_magnitude

        if not correction_is_reasonable(correction):
            print("Turn {} skipped: correction {:.3f} looks unreasonable".format(
                index + 1,
                correction,
            ))
            continue

        turn_scale *= correction

        print(
            "Turn {}: requested={:.2f}, measured={:.2f}, correction={:.4f}, new_turn={:.6f}".format(
                index + 1,
                requested_angle,
                measured_angle,
                correction,
                turn_scale,
            )
        )

        if not apply_motion_calibration(sock, turn_scale, drive_scale):
            break

    return turn_scale


def calibrate_drive(sock, camera, turn_scale, drive_scale, args):
    print()
    print("=" * 60)
    print("Drive calibration")
    print("=" * 60)

    iterations = max(1, int(args.iterations))

    for index in range(iterations):
        before_pose = read_pose(camera, "Drive {} before".format(index + 1))

        if before_pose is None:
            break

        distance = choose_safe_drive_distance(
            before_pose,
            args.drive_distance,
            args.margin,
        )

        if distance is None:
            print("Drive {} skipped: no safe straight calibration distance from this pose".format(
                index + 1,
            ))
            break

        distance_units = int(round(distance))

        if not sync_robot_pose_value(sock, before_pose, "Drive {}".format(index + 1)):
            break

        print("Drive {}: requesting {} map units".format(index + 1, distance_units))

        if not send_required_command(
            sock,
            build_raw_drive(distance_units, int(args.drive_speed)),
            "RAW_DRIVE",
        ):
            break

        time.sleep(float(args.settle))
        after_pose = read_pose(camera, "Drive {} after".format(index + 1))

        if after_pose is None:
            break

        measured_distance = pose_distance(before_pose, after_pose)

        if measured_distance < MIN_DRIVE_MEASUREMENT_UNITS:
            print("Drive {} skipped: measured only {:.2f} map units".format(
                index + 1,
                measured_distance,
            ))
            continue

        correction = abs(float(distance_units)) / measured_distance

        if not correction_is_reasonable(correction):
            print("Drive {} skipped: correction {:.3f} looks unreasonable".format(
                index + 1,
                correction,
            ))
            continue

        drive_scale *= correction

        print(
            "Drive {}: requested={}, measured={:.2f}, correction={:.4f}, new_drive={:.6f}".format(
                index + 1,
                distance_units,
                measured_distance,
                correction,
                drive_scale,
            )
        )

        if not apply_motion_calibration(sock, turn_scale, drive_scale):
            break

    return drive_scale


def main():
    args = parse_args()
    saved_turn_scale, saved_drive_scale = load_local_motion_calibration()
    turn_scale = saved_turn_scale if args.turn_scale is None else float(args.turn_scale)
    drive_scale = saved_drive_scale if args.drive_scale is None else float(args.drive_scale)
    camera = None

    print("=" * 60)
    print("Motion Calibration")
    print("=" * 60)
    print("Host: {}:{}".format(args.host, args.port))
    print("Camera index:", args.camera)
    print("Starting turn scale: {:.6f}".format(turn_scale))
    print("Starting drive scale: {:.6f}".format(drive_scale))
    print("Keep the robot clear of balls, walls, and the red cross while this runs.")
    print()

    try:
        camera = open_camera(args.camera)

        if camera is None:
            print("ERROR: Cannot open camera {}".format(args.camera))
            return 1

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((args.host, args.port))

            if not send_required_command(sock, build_handshake(), "HANDSHAKE"):
                return 1

            if not send_required_command(
                sock,
                build_mapsize(EV3_MAP_HEIGHT, EV3_MAP_WIDTH),
                "MAPSIZE",
            ):
                return 1

            if not apply_motion_calibration(sock, turn_scale, drive_scale):
                return 1

            if not args.skip_turn:
                turn_scale = calibrate_turn(sock, camera, turn_scale, drive_scale, args)
                if not apply_motion_calibration(sock, turn_scale, drive_scale):
                    return 1

            if not args.skip_drive:
                drive_scale = calibrate_drive(sock, camera, turn_scale, drive_scale, args)
                if not apply_motion_calibration(sock, turn_scale, drive_scale):
                    return 1

        save_local_motion_calibration(turn_scale, drive_scale)

        print()
        print("=" * 60)
        print("Motion calibration complete")
        print("=" * 60)
        print("Final turn scale: {:.6f}".format(turn_scale))
        print("Final drive scale: {:.6f}".format(drive_scale))
        return 0

    except OSError as exc:
        print("Connection/camera error:", exc)
        return 1

    finally:
        close_camera(camera)


if __name__ == "__main__":
    raise SystemExit(main())
