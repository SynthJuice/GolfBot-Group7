#!/usr/bin/env python3
import socket
import sys
import time

from camera import (
    close_camera,
    open_camera,
    show_camera_once,
)
from robot_sync import get_robot_pose_from_camera, sync_robot_pose_value
from settings import CAMERA_INDEX, EV3_MAP_HEIGHT, EV3_MAP_WIDTH, SYNC_DELAY_SECONDS
from com_protocol import (
    HOST,
    PORT,
    build_calibrate,
    build_finish,
    build_goto,
    build_handshake,
    build_mapsize,
    build_possync,
    build_setspeed,
    build_turn,
    build_claw_action,
    send_command,
)


def sync_robot_from_camera(sock, camera):
    pose = get_robot_pose_from_camera(camera)

    if pose is None:
        print("Could not detect robot pose from camera")
        return True

    return sync_robot_pose_value(sock, pose, label="Camera")


def goto_xy_then_sync(sock, camera, x, y):
    x = int(x)
    y = int(y)

    print("Sending GOTO x={}, y={}".format(x, y))

    if not send_command(sock, build_goto(x, y)):
        return False

    time.sleep(SYNC_DELAY_SECONDS)

    return sync_robot_from_camera(sock, camera)


def print_help():
    print()
    print("Commands:")
    print("  goto X Y              drive to X,Y, then sync from camera automatically")
    print("  sync                  sync EV3 position from camera now")
    print("  raw_goto X Y          drive to X,Y without camera sync")
    print("  possync X Y HEADING   manually sync pose, heading in tenths of a degree")
    print("  turn ANGLE SPEED")
    print("  setspeed LEFT RIGHT")
    print("  stop")
    print("  claw open")
    print("  claw close")
    print("  claw stop")
    print("  claw deliver")
    print("  claw corner")
    print("  calibrate LEFT_TRIM RIGHT_TRIM")
    print("  handshake")
    print("  mapsize               send current PC map dimensions to the EV3")
    print("  preview               show one prepared camera frame")
    print("  finish")
    print("  help")
    print("  quit")
    print()
    print("Notes:")
    print("  goto uses EV3 coordinates directly: x = column, y = row")
    print("  after each goto, this program asks camera.py for the camera pose and sends POSSYNC")
    print()


def interactive_loop(sock, camera):
    print_help()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd == "goto":
                if len(parts) != 3:
                    print("Usage: goto X Y")
                    continue

                x = int(parts[1])
                y = int(parts[2])

                if not goto_xy_then_sync(sock, camera, x, y):
                    break

            elif cmd == "sync":
                if not sync_robot_from_camera(sock, camera):
                    break

            elif cmd == "raw_goto":
                if len(parts) != 3:
                    print("Usage: raw_goto X Y")
                    continue

                x = int(parts[1])
                y = int(parts[2])

                if not send_command(sock, build_goto(x, y)):
                    break

            elif cmd == "possync":
                if len(parts) != 4:
                    print("Usage: possync X Y HEADING_TENTHS")
                    continue

                x = int(parts[1])
                y = int(parts[2])
                heading_tenths = int(parts[3])

                if not send_command(sock, build_possync(x, y, heading_tenths)):
                    break

            elif cmd == "turn":
                if len(parts) != 3:
                    print("Usage: turn ANGLE SPEED")
                    continue

                angle = int(parts[1])
                speed = int(parts[2])

                if not send_command(sock, build_turn(angle, speed)):
                    break

            elif cmd == "setspeed":
                if len(parts) != 3:
                    print("Usage: setspeed LEFT RIGHT")
                    continue

                left = int(parts[1])
                right = int(parts[2])

                if not send_command(sock, build_setspeed(left, right)):
                    break

            elif cmd == "stop":
                if not send_command(sock, build_setspeed(0, 0)):
                    break

            elif cmd == "claw":
                if len(parts) != 2:
                    print("Usage: claw open")
                    print("   or: claw close")
                    print("   or: claw stop")
                    print("   or: claw deliver")
                    print("   or: claw corner")
                    continue

                packet = build_claw_action(parts[1])

                if packet is None:
                    print("Usage: claw open")
                    print("   or: claw close")
                    print("   or: claw stop")
                    print("   or: claw deliver")
                    print("   or: claw corner")
                    continue

                if not send_command(sock, packet):
                    break

            elif cmd == "calibrate":
                if len(parts) != 3:
                    print("Usage: calibrate LEFT_TRIM RIGHT_TRIM")
                    continue

                left_trim = int(parts[1])
                right_trim = int(parts[2])

                if not send_command(sock, build_calibrate(left_trim, right_trim)):
                    break

            elif cmd == "handshake":
                if not send_command(sock, build_handshake()):
                    break

            elif cmd == "mapsize":
                if not send_command(sock, build_mapsize(EV3_MAP_HEIGHT, EV3_MAP_WIDTH)):
                    break

            elif cmd == "preview":
                show_camera_once(camera)

            elif cmd == "finish":
                if not send_command(sock, build_finish()):
                    break

            elif cmd == "help":
                print_help()

            elif cmd in ("quit", "exit"):
                break

            else:
                print("Unknown command. Type 'help'.")

        except ValueError as exc:
            print("Invalid input:", exc)


def main():
    host = HOST
    port = PORT
    camera_index = CAMERA_INDEX

    if len(sys.argv) >= 2:
        host = sys.argv[1]

    if len(sys.argv) >= 3:
        port = int(sys.argv[2])

    if len(sys.argv) >= 4:
        camera_index = int(sys.argv[3])

    camera = open_camera(camera_index)

    if camera is None:
        return

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))

            print("Connected to EV3 at {}:{}".format(host, port))

            if not send_command(sock, build_handshake()):
                return
            if not send_command(sock, build_mapsize(EV3_MAP_HEIGHT, EV3_MAP_WIDTH)):
                return

            interactive_loop(sock, camera)

    except OSError as exc:
        print("Connection error:", exc)

    finally:
        close_camera(camera)


if __name__ == "__main__":
    main()
