#!/usr/bin/env python3
import socket
import struct
import sys

try:
    from settings import EV3_MAP_HEIGHT, EV3_MAP_WIDTH
except ImportError:
    EV3_MAP_HEIGHT = 480
    EV3_MAP_WIDTH = 640


HOST = "ev3dev.local"
PORT = 5000

ERROR = 0x0
CALIBRATE = 0x1
SENDMAP = 0x2
RAW_TURN = 0x3
MAPSIZE = 0x4
MOTIONCAL = 0x5
RAW_DRIVE = 0x6
CLAW = 0x9
HANDSHAKE = 0xA
GOTO = 0xB
POSSYNC = 0xC
TURN = 0xD
SETSPEED = 0xE
FINISH = 0xF

CLAW_CLOSE = 0x0
CLAW_OPEN = 0x1
CLAW_STOP = 0x2
CLAW_DELIVER = 0x3
CLAW_CORNER = 0x4

CLAW_ACTIONS = {
    "close": CLAW_CLOSE,
    "open": CLAW_OPEN,
    "stop": CLAW_STOP,
    "deliver": CLAW_DELIVER,
    "corner": CLAW_CORNER,
    "corner_ball": CLAW_CORNER,
    "pickup_corner": CLAW_CORNER,
}

MOTION_CALIBRATION_SCALE = 10000.0


def validate_int_range(value, name, lower, upper):
    if not (lower <= value <= upper):
        raise ValueError("{} must be between {} and {}".format(name, lower, upper))
    return value


def encode_signed_byte(value):
    validate_int_range(value, "value", -128, 127)
    return value % 256


def validate_signed_byte(value, name):
    return validate_int_range(value, name, -128, 127)


def validate_signed_short(value, name):
    return validate_int_range(value, name, -32768, 32767)


def validate_signed_int(value, name):
    return validate_int_range(value, name, -2147483648, 2147483647)


def validate_unsigned_short(value, name):
    return validate_int_range(value, name, 0, 65535)


def validate_unsigned_byte(value, name):
    return validate_int_range(value, name, 0, 255)


def encode_motion_calibration_value(value, name):
    value = float(value)

    if value <= 0.0:
        raise ValueError("{} must be positive".format(name))

    return validate_signed_int(
        int(round(value * MOTION_CALIBRATION_SCALE)),
        name,
    )


def packet_preview(packet, max_bytes=40):
    preview = list(packet[:max_bytes])
    if len(packet) > max_bytes:
        return "{} ... total {} bytes".format(preview, len(packet))
    return "{} total {} bytes".format(preview, len(packet))


def recv_response(sock):
    try:
        data = sock.recv(1024)
        if not data:
            print("Robot closed the connection")
            return False

        response = data.decode("utf-8", errors="replace").strip()
        print("Robot:", response)

        if response.startswith("ERR"):
            return False

        return True

    except OSError as exc:
        print("Receive error:", exc)
        return False


def _signed_packet_byte(value):
    return value - 256 if value > 127 else value


def _packet_moves_robot(packet):
    if not packet:
        return False

    command = packet[0]

    if command == GOTO:
        return True

    if command == TURN:
        if len(packet) < 4:
            return True
        angle = struct.unpack(">h", packet[1:3])[0]
        return angle != 0

    if command == RAW_TURN:
        if len(packet) < 4:
            return True
        motor_degrees = struct.unpack(">H", packet[1:3])[0]
        return motor_degrees != 0

    if command == RAW_DRIVE:
        if len(packet) < 4:
            return True
        distance = struct.unpack(">h", packet[1:3])[0]
        return distance != 0

    if command == SETSPEED:
        if len(packet) < 3:
            return True
        left = _signed_packet_byte(packet[1])
        right = _signed_packet_byte(packet[2])
        return left != 0 or right != 0

    if command == CLAW and len(packet) >= 2:
        return packet[1] in (CLAW_DELIVER, CLAW_CORNER)

    return False


def _mark_debug_capture_robot_moved(packet):
    if not _packet_moves_robot(packet):
        return

    try:
        from vision_debug_capture import mark_robot_moved
    except ImportError:
        return

    mark_robot_moved("command 0x{:X}".format(packet[0]))


def send_command(sock, packet):
    try:
        sock.sendall(packet)
        print("Sent:", packet_preview(packet))
        ok = recv_response(sock)

        if ok:
            _mark_debug_capture_robot_moved(packet)

        return ok

    except OSError as exc:
        print("Send error:", exc)
        return False


def build_handshake():
    return bytes([HANDSHAKE])


def build_mapsize(rows, cols):
    rows = validate_unsigned_short(rows, "rows")
    cols = validate_unsigned_short(cols, "cols")

    return struct.pack(">BHH", MAPSIZE, rows, cols)


def build_calibrate(left_trim, right_trim):
    left_trim = validate_signed_byte(left_trim, "left_trim")
    right_trim = validate_signed_byte(right_trim, "right_trim")

    return bytes([
        CALIBRATE,
        encode_signed_byte(left_trim),
        encode_signed_byte(right_trim),
    ])


def build_goto(x, y):
    x = validate_signed_int(x, "x")
    y = validate_signed_int(y, "y")

    return struct.pack(">Bii", GOTO, x, y)


def build_possync(x, y, heading_tenths=0):
    x = validate_signed_int(x, "x")
    y = validate_signed_int(y, "y")
    heading_tenths = validate_signed_short(heading_tenths, "heading_tenths")

    return struct.pack(">Biih", POSSYNC, x, y, heading_tenths)


def build_turn(angle, speed):
    angle = validate_signed_short(angle, "angle")
    speed = validate_signed_byte(speed, "speed")

    return struct.pack(">Bhb", TURN, angle, speed)


def build_raw_turn(motor_degrees, speed):
    motor_degrees = validate_unsigned_short(motor_degrees, "motor_degrees")
    speed = validate_unsigned_byte(speed, "speed")

    return struct.pack(">BHB", RAW_TURN, motor_degrees, speed)


def build_motioncal(degrees_per_turn_degree, degrees_per_map_unit):
    turn_scaled = encode_motion_calibration_value(
        degrees_per_turn_degree,
        "degrees_per_turn_degree",
    )
    drive_scaled = encode_motion_calibration_value(
        degrees_per_map_unit,
        "degrees_per_map_unit",
    )

    return struct.pack(">Bii", MOTIONCAL, turn_scaled, drive_scaled)


def build_raw_drive(distance_map_units, speed=0):
    distance_map_units = validate_signed_short(distance_map_units, "distance_map_units")
    speed = validate_signed_byte(speed, "speed")

    return struct.pack(">Bhb", RAW_DRIVE, distance_map_units, speed)


def build_setspeed(left, right):
    left = validate_signed_byte(left, "left")
    right = validate_signed_byte(right, "right")

    return bytes([
        SETSPEED,
        encode_signed_byte(left),
        encode_signed_byte(right),
    ])


def build_claw(action):
    action = validate_unsigned_byte(action, "action")

    if action not in (CLAW_CLOSE, CLAW_OPEN, CLAW_STOP, CLAW_DELIVER, CLAW_CORNER):
        raise ValueError("claw action must be 0, 1, 2, 3, or 4")

    return bytes([CLAW, action])


def build_claw_action(action):
    action_code = CLAW_ACTIONS.get(action.lower())

    if action_code is None:
        return None

    return build_claw(action_code)


def build_claw_open():
    return build_claw(CLAW_OPEN)


def build_claw_close():
    return build_claw(CLAW_CLOSE)


def build_claw_stop():
    return build_claw(CLAW_STOP)


def build_claw_deliver():
    return build_claw(CLAW_DELIVER)


def build_claw_corner():
    return build_claw(CLAW_CORNER)


def build_sendmap(rows, cols, cells):
    rows = validate_unsigned_short(rows, "rows")
    cols = validate_unsigned_short(cols, "cols")

    expected = rows * cols

    if len(cells) != expected:
        raise ValueError(
            "sendmap needs exactly {} cell values for a {}x{} map".format(
                expected, rows, cols
            )
        )

    validated_cells = [validate_unsigned_byte(cell, "cell") for cell in cells]

    return struct.pack(">BHH", SENDMAP, rows, cols) + bytes(validated_cells)


def build_sendmap_fill(rows, cols, value):
    rows = validate_unsigned_short(rows, "rows")
    cols = validate_unsigned_short(cols, "cols")
    value = validate_unsigned_byte(value, "value")

    cells = bytes([value]) * (rows * cols)

    return struct.pack(">BHH", SENDMAP, rows, cols) + cells


def build_finish():
    return bytes([FINISH])


def print_help():
    print()
    print("Commands:")
    print("  handshake")
    print("  mapsize")
    print("  mapsize ROWS COLS")
    print("  calibrate LEFT_TRIM RIGHT_TRIM")
    print("  goto X Y")
    print("  possync X Y")
    print("  possync X Y HEADING_TENTHS")
    print("  turn ANGLE SPEED")
    print("  raw_turn MOTOR_DEGREES SPEED")
    print("  motioncal DEGREES_PER_TURN_DEGREE DEGREES_PER_MAP_UNIT")
    print("  raw_drive DISTANCE_MAP_UNITS SPEED")
    print("  setspeed LEFT RIGHT")
    print("  claw open")
    print("  claw close")
    print("  claw stop")
    print("  claw deliver")
    print("  claw corner")
    print("  open_claw")
    print("  close_claw")
    print("  stop_claw")
    print("  deliver_ball")
    print("  corner_ball")
    print("  sendmap ROWS COLS CELL1 CELL2 ...")
    print("  sendmap_fill ROWS COLS VALUE")
    print("  finish")
    print("  help")
    print("  quit")
    print()
    print("Notes:")
    print("  goto uses signed 32-bit x/y")
    print("  possync uses signed 32-bit x/y and signed 16-bit heading_tenths")
    print("  possync packet length is 11 bytes")
    print("  turn angle uses signed 16-bit integer")
    print("  turn speed uses signed byte: -128..127")
    print("  raw_turn motor degrees use unsigned 16-bit integer")
    print("  motioncal values are positive floats")
    print("  raw_drive distance uses signed 16-bit integer")
    print("  setspeed/calibrate use signed bytes: -128..127")
    print("  claw action is sent as: 0=close, 1=open, 2=stop, 3=deliver, 4=corner pickup")
    print("  sendmap rows/cols use unsigned 16-bit integers")
    print("  sendmap cell values are bytes: 0..255")
    print()


def interactive_loop(sock, host, port):
    print("Connected to EV3 at {}:{}".format(host, port))
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
            if cmd == "handshake":
                packet = build_handshake()

            elif cmd == "mapsize":
                if len(parts) == 1:
                    rows = EV3_MAP_HEIGHT
                    cols = EV3_MAP_WIDTH
                elif len(parts) == 3:
                    rows = int(parts[1])
                    cols = int(parts[2])
                else:
                    print("Usage: mapsize")
                    print("   or: mapsize ROWS COLS")
                    continue

                packet = build_mapsize(rows, cols)

            elif cmd == "calibrate":
                if len(parts) != 3:
                    print("Usage: calibrate LEFT_TRIM RIGHT_TRIM")
                    continue

                left_trim = int(parts[1])
                right_trim = int(parts[2])

                packet = build_calibrate(left_trim, right_trim)

            elif cmd == "goto":
                if len(parts) != 3:
                    print("Usage: goto X Y")
                    continue

                x = int(parts[1])
                y = int(parts[2])

                packet = build_goto(x, y)

            elif cmd == "possync":
                if len(parts) not in (3, 4):
                    print("Usage: possync X Y")
                    print("   or: possync X Y HEADING_TENTHS")
                    continue

                x = int(parts[1])
                y = int(parts[2])

                if len(parts) == 4:
                    heading_tenths = int(parts[3])
                else:
                    heading_tenths = 0

                packet = build_possync(x, y, heading_tenths)

            elif cmd == "turn":
                if len(parts) != 3:
                    print("Usage: turn ANGLE SPEED")
                    continue

                angle = int(parts[1])
                speed = int(parts[2])

                packet = build_turn(angle, speed)

            elif cmd == "raw_turn":
                if len(parts) != 3:
                    print("Usage: raw_turn MOTOR_DEGREES SPEED")
                    continue

                motor_degrees = int(parts[1])
                speed = int(parts[2])

                packet = build_raw_turn(motor_degrees, speed)

            elif cmd == "motioncal":
                if len(parts) != 3:
                    print("Usage: motioncal DEGREES_PER_TURN_DEGREE DEGREES_PER_MAP_UNIT")
                    continue

                degrees_per_turn_degree = float(parts[1])
                degrees_per_map_unit = float(parts[2])

                packet = build_motioncal(degrees_per_turn_degree, degrees_per_map_unit)

            elif cmd == "raw_drive":
                if len(parts) != 3:
                    print("Usage: raw_drive DISTANCE_MAP_UNITS SPEED")
                    continue

                distance_map_units = int(parts[1])
                speed = int(parts[2])

                packet = build_raw_drive(distance_map_units, speed)

            elif cmd == "setspeed":
                if len(parts) != 3:
                    print("Usage: setspeed LEFT RIGHT")
                    continue

                left = int(parts[1])
                right = int(parts[2])

                packet = build_setspeed(left, right)

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

            elif cmd == "open_claw":
                packet = build_claw_open()

            elif cmd == "close_claw":
                packet = build_claw_close()

            elif cmd == "stop_claw":
                packet = build_claw_stop()

            elif cmd == "deliver_ball":
                packet = build_claw_deliver()

            elif cmd == "corner_ball":
                packet = build_claw_corner()

            elif cmd == "sendmap":
                if len(parts) < 4:
                    print("Usage: sendmap ROWS COLS CELL1 CELL2 ...")
                    continue

                rows = int(parts[1])
                cols = int(parts[2])
                cells = [int(value) for value in parts[3:]]

                packet = build_sendmap(rows, cols, cells)

            elif cmd == "sendmap_fill":
                if len(parts) != 4:
                    print("Usage: sendmap_fill ROWS COLS VALUE")
                    continue

                rows = int(parts[1])
                cols = int(parts[2])
                value = int(parts[3])

                packet = build_sendmap_fill(rows, cols, value)

            elif cmd == "finish":
                packet = build_finish()

            elif cmd == "help":
                print_help()
                continue

            elif cmd in ("quit", "exit"):
                break

            else:
                print("Unknown command. Type 'help'.")
                continue

            if not send_command(sock, packet):
                break

        except ValueError as exc:
            print("Invalid input:", exc)


def main():
    host = HOST
    port = PORT

    if len(sys.argv) >= 2:
        host = sys.argv[1]

    if len(sys.argv) >= 3:
        port = int(sys.argv[2])

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            interactive_loop(sock, host, port)

    except OSError as exc:
        print("Connection error:", exc)


if __name__ == "__main__":
    main()
