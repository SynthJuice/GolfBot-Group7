import time

from com_protocol import build_claw_close, build_claw_open, build_setspeed, send_command
from pickup_motion import servo_align_and_approach_ball
from settings import PICKUP_SETTLE_SECONDS


def approach_ball_and_close_claw(
    sock,
    camera,
    ball_color="W",
    open_claw=True,
    target_ball_point=None,
):
    if open_claw:
        print("Opening claw before pickup approach")
        if not send_command(sock, build_claw_open()):
            return False
    else:
        print("Continuing pickup without reopening claw")

    if not servo_align_and_approach_ball(
        sock,
        camera,
        ball_color=ball_color,
        target_ball_point=target_ball_point,
    ):
        print("Pickup servo failed; not closing claw blindly")
        return False

    print("Stopping before closing claw")
    if not send_command(sock, build_setspeed(0, 0)):
        return False

    time.sleep(PICKUP_SETTLE_SECONDS)

    print("Closing claw at pickup point")
    return send_command(sock, build_claw_close())
