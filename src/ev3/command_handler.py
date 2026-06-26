#!/usr/bin/env python3
import struct

from ev3dev2.motor import MediumMotor, SpeedPercent, OUTPUT_B

ERROR     = 0x0
CALIBRATE = 0x1
SENDMAP   = 0x2
RAW_TURN  = 0x3
MAPSIZE   = 0x4
MOTIONCAL = 0x5
RAW_DRIVE = 0x6
CLAW      = 0x9
HANDSHAKE = 0xA
GOTO      = 0xB
POSSYNC   = 0xC
TURN      = 0xD
SETSPEED  = 0xE
FINISH    = 0xF

CLAW_CLOSE   = 0x0
CLAW_OPEN    = 0x1
CLAW_STOP    = 0x2
CLAW_DELIVER = 0x3
CLAW_CORNER  = 0x4

ERROR_LENGTH          = 1
CALIBRATE_LENGTH      = 3
SENDMAP_HEADER_LENGTH = 5
HANDSHAKE_LENGTH      = 1
RAW_TURN_LENGTH       = 4
MAPSIZE_LENGTH        = 5
MOTIONCAL_LENGTH      = 9
RAW_DRIVE_LENGTH      = 4
CLAW_LENGTH           = 2
GOTO_LENGTH           = 9
POSSYNC_LENGTH        = 11
TURN_LENGTH           = 4
SETSPEED_LENGTH       = 3
FINISH_LENGTH         = 1

MOTION_CALIBRATION_SCALE = 10000.0


def byte_to_signed(value):
    if value > 127:
        return value - 256
    return value


def read_int32(data, offset):
    return struct.unpack(">i", data[offset:offset + 4])[0]


def read_int16(data, offset):
    return struct.unpack(">h", data[offset:offset + 2])[0]


def read_uint16(data, offset):
    return struct.unpack(">H", data[offset:offset + 2])[0]


def read_motioncal_value(data, offset):
    return read_int32(data, offset) / MOTION_CALIBRATION_SCALE


def sendmap_length(data):
    if len(data) < SENDMAP_HEADER_LENGTH:
        return None
    rows = read_uint16(data, 1)
    cols = read_uint16(data, 3)
    return SENDMAP_HEADER_LENGTH + (rows * cols)


class Command:
    def __init__(self, code, length, handler):
        self.code    = code
        self.length  = length
        self.handler = handler

    def get_expected_length(self, data):
        if callable(self.length):
            return self.length(data)
        return self.length

    def execute(self, data):
        expected_length = self.get_expected_length(data)

        if expected_length is None:
            print("Command 0x{:X} does not yet have enough header bytes".format(self.code))
            return False

        if len(data) < expected_length:
            print("Invalid length for command 0x{:X}: expected {}, got {}".format(
                self.code, expected_length, len(data)))
            return False

        self.handler(data)
        return True


class CommandHandler:
    def __init__(self, motor_controller):
        self.motor_controller = motor_controller

        self.claw  = MediumMotor(OUTPUT_B)
        self.tank  = motor_controller.tank

        self.map = {"rows": 0, "cols": 0, "cells": []}

        self.commands = {
            ERROR:     Command(ERROR,     ERROR_LENGTH,     self.error_command),
            CALIBRATE: Command(CALIBRATE, CALIBRATE_LENGTH, self.calibrate_command),
            SENDMAP:   Command(SENDMAP,   sendmap_length,   self.sendmap_command),
            RAW_TURN:  Command(RAW_TURN,  RAW_TURN_LENGTH,  self.raw_turn_command),
            MAPSIZE:   Command(MAPSIZE,   MAPSIZE_LENGTH,   self.mapsize_command),
            MOTIONCAL: Command(MOTIONCAL, MOTIONCAL_LENGTH, self.motioncal_command),
            RAW_DRIVE: Command(RAW_DRIVE, RAW_DRIVE_LENGTH, self.raw_drive_command),
            CLAW:      Command(CLAW,      CLAW_LENGTH,      self.claw_command),
            HANDSHAKE: Command(HANDSHAKE, HANDSHAKE_LENGTH, self.handshake),
            GOTO:      Command(GOTO,      GOTO_LENGTH,      self.goto),
            POSSYNC:   Command(POSSYNC,   POSSYNC_LENGTH,   self.position_sync),
            TURN:      Command(TURN,      TURN_LENGTH,      self.turn),
            SETSPEED:  Command(SETSPEED,  SETSPEED_LENGTH,  self.set_speed),
            FINISH:    Command(FINISH,    FINISH_LENGTH,    self.finish_command),
        }

    def handshake(self, data):
        print("HANDSHAKE")

    def goto(self, data):
        x = read_int32(data, 1)
        y = read_int32(data, 5)
        print("GOTO ({}, {})".format(x, y))
        self.motor_controller.goto(x, y)

    def position_sync(self, data):
        x       = read_int32(data, 1)
        y       = read_int32(data, 5)
        heading = read_int16(data, 9) / 10.0
        print("POSSYNC ({}, {}), heading={:.1f}".format(x, y, heading))
        self.motor_controller.set_position(x, y)
        self.motor_controller.set_heading(heading)

    def turn(self, data):
        angle = read_int16(data, 1)
        speed = byte_to_signed(data[3])
        print("TURN angle={}, speed={}".format(angle, speed))
        self.motor_controller.turn(angle, speed)

    def set_speed(self, data):
        left  = byte_to_signed(data[1])
        right = byte_to_signed(data[2])
        self.motor_controller.set_speed(left, right)

    def error_command(self, data):
        print("ERROR COMMAND RECEIVED")
        self.motor_controller.stop()

    def calibrate_command(self, data):
        left_trim  = byte_to_signed(data[1])
        right_trim = byte_to_signed(data[2])
        self.motor_controller.calibrate(left_trim, right_trim)

    def finish_command(self, data):
        print("FINISH COMMAND")
        self.motor_controller.stop()

    def sendmap_command(self, data):
        rows = read_uint16(data, 1)
        cols = read_uint16(data, 3)
        expected_cells = rows * cols
        raw_cells = list(data[SENDMAP_HEADER_LENGTH:SENDMAP_HEADER_LENGTH + expected_cells])
        cells_2d = []
        for r in range(rows):
            start = r * cols
            end   = start + cols
            cells_2d.append(raw_cells[start:end])
        self.map = {"rows": rows, "cols": cols, "cells": cells_2d}
        self.motor_controller.set_map_dimensions(rows, cols)
        print("SENDMAP rows={}, cols={}, cells={}".format(rows, cols, expected_cells))

    def mapsize_command(self, data):
        rows = read_uint16(data, 1)
        cols = read_uint16(data, 3)
        self.motor_controller.set_map_dimensions(rows, cols)
        print("MAPSIZE rows={}, cols={}".format(rows, cols))

    def motioncal_command(self, data):
        degrees_per_turn_degree = read_motioncal_value(data, 1)
        degrees_per_map_unit = read_motioncal_value(data, 5)

        print("MOTIONCAL turn={:.4f}, drive={:.4f}".format(
            degrees_per_turn_degree,
            degrees_per_map_unit,
        ))
        self.motor_controller.set_motion_calibration(
            degrees_per_turn_degree,
            degrees_per_map_unit,
            persist=True,
        )

    def raw_drive_command(self, data):
        distance_map_units = read_int16(data, 1)
        speed = byte_to_signed(data[3])

        print("RAW_DRIVE distance={}, speed={}".format(distance_map_units, speed))
        self.motor_controller.drive_straight(distance_map_units, speed)

    def raw_turn_command(self, data):
        motor_degrees = struct.unpack(">H", data[1:3])[0]
        speed         = data[3]

        print("RAW_TURN motor_degrees={}, speed={}".format(motor_degrees, speed))

        mc   = self.motor_controller
        spd  = float(min(max(speed, 1), 100))

        left_cmd  = spd  * mc.LEFT_MOTOR_DIRECTION
        right_cmd = -spd * mc.RIGHT_MOTOR_DIRECTION

        left_cmd  = max(-100.0, min(100.0, float(left_cmd)))
        right_cmd = max(-100.0, min(100.0, float(right_cmd)))

        mc.tank.on_for_degrees(
            SpeedPercent(left_cmd),
            SpeedPercent(right_cmd),
            motor_degrees,
            brake=True,
            block=True,
        )
        print("RAW_TURN done")

    def claw_command(self, data):
        action = data[1]
        print("CLAW action={}".format(action))

        if action == CLAW_OPEN:
            self._claw_open()

        elif action == CLAW_CLOSE:
            self._claw_close()

        elif action == CLAW_STOP:
            self.claw.off(brake=True)
            print("CLAW STOP")

        elif action == CLAW_DELIVER:
            self._claw_deliver()

        elif action == CLAW_CORNER:
            self._claw_corner()

        else:
            print("CLAW: unknown action {}".format(action))

    def _claw_open(self):
        print("CLAW OPEN")
        self.claw.on_for_rotations(100, 3.5, block=False)

    def _claw_close(self):
        print("CLAW CLOSE")
        self.claw.on_for_rotations(-100, 3.5, block=True)

    def _claw_deliver(self):
        print("CLAW DELIVER")
        self._claw_open()
        self.tank.on_for_rotations(-42, -40, 1, brake=False)
        self.tank.on_for_rotations( 42,  40, 1, brake=False)
        self._claw_close()

    def _claw_corner(self):
        print("CLAW CORNER PICKUP")
        for _ in range(5):
            self.tank.on_for_rotations(20, 20, 0.1, brake=True)
            self.claw.on_for_rotations(20, 0.2, brake=True)
        self.claw.on_for_rotations(30, 0.5, brake=True)
        self.tank.on_for_rotations(-30, -30, 0.6, brake=True)

    def get_expected_length(self, data):
        if not data:
            return None
        cmd_code = data[0]
        command  = self.commands.get(cmd_code)
        if command is None:
            return 1
        return command.get_expected_length(data)

    def handle_command(self, data):
        if not data:
            return False
        cmd_code = data[0]
        command  = self.commands.get(cmd_code)
        if command is None:
            print("Invalid command received: 0x{:X}".format(cmd_code))
            self.motor_controller.stop()
            return False
        return command.execute(data)
