#!/usr/bin/env python3
import math
import os
from ev3dev2.motor import OUTPUT_A, OUTPUT_D, MoveTank, SpeedPercent


MOTION_CALIBRATION_FILENAME = "motion_calibration.txt"


def _calibration_path(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def _load_degrees_per_turn_degree(default=2.35):
    calib_path = _calibration_path("calibration.txt")
    if os.path.exists(calib_path):
        try:
            with open(calib_path, "r") as calibration_file:
                value = float(calibration_file.read().strip())
            print("[motor_control] Loaded DEGREES_PER_TURN_DEGREE={:.4f} from {}".format(
                value, calib_path))
            return value
        except (ValueError, OSError) as exc:
            print("[motor_control] WARNING: Could not read {}: {}. Using default {}.".format(
                calib_path, exc, default))
    else:
        print("[motor_control] WARNING: No calibration.txt found at {}. "
              "Using default DEGREES_PER_TURN_DEGREE={}. "
              "Motion calibration can update it later.".format(
                  calib_path, default))
    return default


def _load_motion_calibration(default_turn, default_drive):
    calib_path = _calibration_path(MOTION_CALIBRATION_FILENAME)
    turn = float(default_turn)
    drive = float(default_drive)

    if not os.path.exists(calib_path):
        return turn, drive

    try:
        with open(calib_path, "r") as calibration_file:
            lines = calibration_file.readlines()

        for line in lines:
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            if "=" in stripped:
                name, value = stripped.split("=", 1)
                name = name.strip().lower()
                value = float(value.strip())

                if name in ("degrees_per_turn_degree", "turn"):
                    turn = value
                elif name in ("degrees_per_map_unit", "drive"):
                    drive = value

            else:
                parts = stripped.split()

                if len(parts) >= 2:
                    turn = float(parts[0])
                    drive = float(parts[1])

        print("[motor_control] Loaded motion calibration turn={:.4f}, drive={:.4f} from {}".format(
            turn, drive, calib_path))

    except (ValueError, OSError) as exc:
        print("[motor_control] WARNING: Could not read {}: {}. Using defaults.".format(
            calib_path, exc))
        return float(default_turn), float(default_drive)

    if turn <= 0.0 or drive <= 0.0:
        print("[motor_control] WARNING: Invalid motion calibration turn={:.4f}, drive={:.4f}. "
              "Using defaults.".format(turn, drive))
        return float(default_turn), float(default_drive)

    return turn, drive


def _save_motion_calibration(degrees_per_turn_degree, degrees_per_map_unit):
    calib_path = _calibration_path(MOTION_CALIBRATION_FILENAME)

    try:
        with open(calib_path, "w") as calibration_file:
            calibration_file.write("degrees_per_turn_degree={:.6f}\n".format(
                float(degrees_per_turn_degree)))
            calibration_file.write("degrees_per_map_unit={:.6f}\n".format(
                float(degrees_per_map_unit)))
        print("[motor_control] Saved motion calibration to {}".format(calib_path))
        return True

    except OSError as exc:
        print("[motor_control] WARNING: Could not save {}: {}".format(calib_path, exc))
        return False


class MotorController:
    DEFAULT_TURN_SPEED  = 20
    DEFAULT_DRIVE_SPEED = 30

    MAP_COLS = 640
    MAP_ROWS = 480

    DEFAULT_CM_PER_MAP_UNIT = 0.26
    DEGREES_PER_CM = 20

    DEFAULT_DEGREES_PER_TURN_DEGREE = _load_degrees_per_turn_degree(default=2.35)
    DEFAULT_DEGREES_PER_MAP_UNIT = DEGREES_PER_CM * DEFAULT_CM_PER_MAP_UNIT

    DEGREES_PER_TURN_DEGREE, DEGREES_PER_MAP_UNIT = _load_motion_calibration(
        DEFAULT_DEGREES_PER_TURN_DEGREE,
        DEFAULT_DEGREES_PER_MAP_UNIT,
    )
    CM_PER_MAP_UNIT = DEGREES_PER_MAP_UNIT / float(DEGREES_PER_CM)

    LEFT_MOTOR_DIRECTION  = -1
    RIGHT_MOTOR_DIRECTION = -1

    TURN_BRAKE  = False
    DRIVE_BRAKE = False

    def __init__(self, left_output=OUTPUT_A, right_output=OUTPUT_D):
        self.tank = MoveTank(left_output, right_output)

        self.left_trim  = 0.0
        self.right_trim = 0.0

        self.x       = float(self.MAP_COLS) / 2.0
        self.y       = float(self.MAP_ROWS) / 2.0
        self.heading = 0.0

    def clamp_speed(self, value):
        return max(-100.0, min(100.0, float(value)))

    def apply_trim(self, speed, trim):
        speed, trim = float(speed), float(trim)
        if speed > 0.0:
            return self.clamp_speed(speed + trim)
        if speed < 0.0:
            return self.clamp_speed(speed - trim)
        return 0.0

    def apply_calibration(self, left, right):
        return (self.apply_trim(left, self.left_trim),
                self.apply_trim(right, self.right_trim))

    def apply_motor_direction(self, left, right):
        return (self.clamp_speed(float(left)  * self.LEFT_MOTOR_DIRECTION),
                self.clamp_speed(float(right) * self.RIGHT_MOTOR_DIRECTION))

    def normalize_heading(self, angle):
        return float(angle) % 360.0

    def normalize_turn(self, angle):
        return (float(angle) + 180.0) % 360.0 - 180.0

    def position_is_valid(self, x, y):
        return (0.0 <= float(x) < float(self.MAP_COLS) and
                0.0 <= float(y) < float(self.MAP_ROWS))

    def set_map_dimensions(self, rows, cols):
        rows = int(rows)
        cols = int(cols)

        if rows <= 0 or cols <= 0:
            print("MAP DIMENSIONS ERROR: rows={}, cols={}".format(rows, cols))
            return

        self.MAP_ROWS = rows
        self.MAP_COLS = cols
        print("MAP DIMENSIONS set to cols={}, rows={}".format(self.MAP_COLS, self.MAP_ROWS))

    def set_heading(self, heading):
        self.heading = self.normalize_heading(heading)
        print("HEADING set to {:.2f}".format(self.heading))

    def set_position(self, x, y):
        x, y = float(x), float(y)
        if not self.position_is_valid(x, y):
            print("POSITION ERROR: ({:.1f}, {:.1f}) is outside map bounds "
                  "(x: 0..{}, y: 0..{})".format(
                      x, y, self.MAP_COLS - 1, self.MAP_ROWS - 1))
            return
        self.x = x
        self.y = y
        print("POSITION set to x={:.2f}, y={:.2f}".format(self.x, self.y))

    def stop(self, brake=True):
        self.tank.off(brake=brake)

    def calibrate(self, left_trim, right_trim):
        self.left_trim  = self.clamp_speed(left_trim)
        self.right_trim = self.clamp_speed(right_trim)
        print("CALIBRATE  left_trim={:.2f}  right_trim={:.2f}".format(
            self.left_trim, self.right_trim))

    def set_motion_calibration(self, degrees_per_turn_degree, degrees_per_map_unit, persist=True):
        degrees_per_turn_degree = float(degrees_per_turn_degree)
        degrees_per_map_unit = float(degrees_per_map_unit)

        if degrees_per_turn_degree <= 0.0 or degrees_per_map_unit <= 0.0:
            print("MOTION CALIBRATION ERROR: turn={:.4f}, drive={:.4f}".format(
                degrees_per_turn_degree,
                degrees_per_map_unit,
            ))
            return False

        self.DEGREES_PER_TURN_DEGREE = degrees_per_turn_degree
        self.DEGREES_PER_MAP_UNIT = degrees_per_map_unit
        self.CM_PER_MAP_UNIT = degrees_per_map_unit / float(self.DEGREES_PER_CM)

        print("MOTION CALIBRATION set: turn={:.4f}, drive={:.4f}, cm_per_map_unit={:.4f}".format(
            self.DEGREES_PER_TURN_DEGREE,
            self.DEGREES_PER_MAP_UNIT,
            self.CM_PER_MAP_UNIT,
        ))

        if persist:
            return _save_motion_calibration(
                self.DEGREES_PER_TURN_DEGREE,
                self.DEGREES_PER_MAP_UNIT,
            )

        return True

    def turn(self, angle, speed=0):
        angle = float(angle)
        if angle == 0.0:
            self.stop()
            return

        speed = abs(self.clamp_speed(float(speed) if speed != 0.0
                                     else float(self.DEFAULT_TURN_SPEED)))
        motor_degrees = abs(angle) * float(self.DEGREES_PER_TURN_DEGREE)

        print("TURN  requested={:.2f}  speed={:.1f}%  motor_degrees={:.2f}".format(
            angle, speed, motor_degrees))

        left_cmd  = -speed if angle > 0.0 else  speed
        right_cmd =  speed if angle > 0.0 else -speed

        left_cmd,  right_cmd = self.apply_calibration(left_cmd, right_cmd)
        left_cmd,  right_cmd = self.apply_motor_direction(left_cmd, right_cmd)

        self.tank.on_for_degrees(
            SpeedPercent(left_cmd),
            SpeedPercent(right_cmd),
            motor_degrees,
            brake=self.TURN_BRAKE,
            block=True,
        )

        self.heading = self.normalize_heading(self.heading + angle)
        print("HEADING now {:.2f}".format(self.heading))

    def drive_straight(self, distance_map_units, speed=0):
        distance_map_units = float(distance_map_units)
        if distance_map_units == 0.0:
            self.stop()
            return

        speed = abs(self.clamp_speed(float(speed) if speed != 0.0
                                     else float(self.DEFAULT_DRIVE_SPEED)))
        distance_cm   = distance_map_units * float(self.CM_PER_MAP_UNIT)
        motor_degrees = abs(distance_map_units) * float(self.DEGREES_PER_MAP_UNIT)

        left_cmd  = speed  if distance_map_units > 0.0 else -speed
        right_cmd = speed  if distance_map_units > 0.0 else -speed

        left_cmd,  right_cmd = self.apply_calibration(left_cmd, right_cmd)
        left_cmd,  right_cmd = self.apply_motor_direction(left_cmd, right_cmd)

        print("DRIVE  dist_units={:.2f}  dist_cm={:.2f}  "
              "speed={:.1f}%  heading={:.2f}  motor_degrees={:.2f}".format(
                  distance_map_units, distance_cm,
                  speed, self.heading, motor_degrees))

        self.tank.on_for_degrees(
            SpeedPercent(left_cmd),
            SpeedPercent(right_cmd),
            motor_degrees,
            brake=self.DRIVE_BRAKE,
            block=True,
        )

        heading_rad = math.radians(self.heading)
        self.x     += distance_map_units * math.cos(heading_rad)
        self.y     += distance_map_units * math.sin(heading_rad)

        print("POSITION now x={:.2f}, y={:.2f}".format(self.x, self.y))

    def goto(self, target_x, target_y, turn_speed=0, drive_speed=0):
        target_x = float(target_x)
        target_y = float(target_y)

        if not self.position_is_valid(target_x, target_y):
            print("GOTO ERROR: ({:.1f}, {:.1f}) is outside map bounds "
                  "(x: 0..{}, y: 0..{})".format(
                      target_x, target_y,
                      self.MAP_COLS - 1, self.MAP_ROWS - 1))
            return

        dx = target_x - self.x
        dy = target_y - self.y

        if dx == 0.0 and dy == 0.0:
            print("GOTO: already at target ({:.1f}, {:.1f})".format(
                target_x, target_y))
            return

        target_heading = self.normalize_heading(math.degrees(math.atan2(dy, dx)))
        turn_angle     = self.normalize_turn(target_heading - self.heading)
        distance       = math.hypot(dx, dy)

        print("GOTO  target=({:.1f}, {:.1f})  current=({:.1f}, {:.1f})  "
              "heading={:.1f}  target_heading={:.1f}  "
              "turn={:.1f}  dist_units={:.1f}  dist_cm={:.1f}".format(
                  target_x, target_y, self.x, self.y,
                  self.heading, target_heading,
                  turn_angle, distance,
                  distance * self.CM_PER_MAP_UNIT))

        self.turn(turn_angle, turn_speed)
        self.drive_straight(distance, drive_speed)

        self.x = target_x
        self.y = target_y
        print("POSITION snapped to x={:.2f}, y={:.2f}".format(self.x, self.y))

    def set_speed(self, left, right):
        left  = self.clamp_speed(left)
        right = self.clamp_speed(right)

        left,  right = self.apply_calibration(left, right)
        left,  right = self.apply_motor_direction(left, right)

        print("SET SPEED  left={:.2f}  right={:.2f}".format(left, right))

        if left == 0.0 and right == 0.0:
            self.stop()
            return

        self.tank.on(SpeedPercent(left), SpeedPercent(right))
