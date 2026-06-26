#!/usr/bin/env python3

from ev3dev2.motor import MoveTank, MediumMotor, OUTPUT_A, OUTPUT_D, OUTPUT_B

Move_Straight = MoveTank(OUTPUT_D, OUTPUT_A)
claw = MediumMotor(OUTPUT_B)


def open_claw():
    claw.on_for_rotations(100, 3.5, block=True)


def close_claw():
    claw.on_for_rotations(-100, 3.5)


def deliver_ball():
    open_claw()

    Move_Straight.on_for_rotations(-42, -40, 1, brake=False)
    Move_Straight.on_for_rotations(42, 40, 1, brake=False)

    close_claw()


def pick_corner_ball():
    for _ in range(5):
        Move_Straight.on_for_rotations(20, 20, 0.1, brake=True)

        claw.on_for_rotations(20, 0.2, brake=True)

    claw.on_for_rotations(30, 0.5, brake=True)

    Move_Straight.on_for_rotations(-30, -30, 0.6, brake=True)


if __name__ == "__main__":
    deliver_ball()
