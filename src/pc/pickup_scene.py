from map_utils import point_distance
from path_obstacles import create_empty_path_matrix, point_in_obstacle_regions, red_cross_obstacle_regions
from scene_analysis import (
    ball_points_from_sources,
    capture_scene_with_robot_pose_retry,
    capture_vision_scene_frame,
    grappler_point_from_sources,
    robot_pose_from_sources,
)
from settings import (
    PICKUP_RED_CROSS_CLEARANCE_MARGIN,
    PICKUP_TARGET_MATCH_MAX_DISTANCE,
    ROBOT_POSE_RETRY_FRAMES,
)


def planning_matrix_from_scene(scene):
    if scene is None:
        return create_empty_path_matrix()

    path_matrix = scene.get("path_matrix")

    if path_matrix is not None:
        return path_matrix

    return create_empty_path_matrix()


def choose_closest_ball_to_grappler(balls, grappler_point):
    if not balls or grappler_point is None:
        return None

    return min(balls, key=lambda ball: point_distance(ball, grappler_point))


def choose_pickup_ball(balls, grappler_point, target_point=None):
    if not balls:
        return None, "none"

    if target_point is not None:
        matched_ball = min(balls, key=lambda ball: point_distance(ball, target_point))
        match_distance = point_distance(matched_ball, target_point)
        max_match_distance = max(0.0, float(PICKUP_TARGET_MATCH_MAX_DISTANCE))

        if match_distance <= max_match_distance:
            print(
                "Pickup camera: following selected target {}; matched visible ball {} "
                "(target error {:.1f})".format(
                    target_point,
                    matched_ball,
                    match_distance,
                )
            )
            return matched_ball, "target"

        print(
            "Pickup camera: selected target {} is not visible within {:.1f} map units; "
            "closest visible ball {} is {:.1f} away, so not switching targets".format(
                target_point,
                max_match_distance,
                matched_ball,
                match_distance,
            )
        )
        return None, "target_missing"

    return choose_closest_ball_to_grappler(balls, grappler_point), "closest"


def filter_balls_for_red_cross_clearance(path_matrix, vision_scene, balls):
    regions = red_cross_obstacle_regions(
        path_matrix,
        vision_scene,
        margin=PICKUP_RED_CROSS_CLEARANCE_MARGIN,
    )

    if not regions:
        return balls, []

    safe_balls = []
    blocked_balls = []

    for ball in balls:
        if point_in_obstacle_regions(ball, regions):
            print(
                "Pickup camera: ignoring ball {} because it is too close to the red cross".format(
                    ball,
                )
            )
            blocked_balls.append(ball)
            continue

        safe_balls.append(ball)

    return safe_balls, blocked_balls


def capture_pickup_scene_frame(camera, ball_color="W", target_point=None):
    scene = capture_vision_scene_frame(camera, "pickup")

    if scene is None:
        return None

    path_matrix = scene["path_matrix"]
    vision_scene = scene["vision_scene"]
    robot_pose = robot_pose_from_sources(vision_scene)
    grappler_point = grappler_point_from_sources(vision_scene)
    balls = ball_points_from_sources(vision_scene, ball_color)

    balls, blocked_balls = filter_balls_for_red_cross_clearance(
        path_matrix,
        vision_scene,
        balls,
    )
    ball_point, ball_selection = choose_pickup_ball(
        balls,
        grappler_point,
        target_point=target_point,
    )

    return {
        "path_matrix": path_matrix,
        "vision_scene": vision_scene,
        "robot_pose": robot_pose,
        "grappler_point": grappler_point,
        "balls": balls,
        "blocked_balls": blocked_balls,
        "pickup_blocked_by_red_cross": bool(blocked_balls and not balls),
        "pickup_target_point": target_point,
        "pickup_target_missing": ball_selection == "target_missing",
        "ball_selection": ball_selection,
        "ball_point": ball_point,
    }


def capture_pickup_scene(
    camera,
    ball_color="W",
    retry_frames=ROBOT_POSE_RETRY_FRAMES,
    target_point=None,
):
    return capture_scene_with_robot_pose_retry(
        lambda: capture_pickup_scene_frame(
            camera,
            ball_color=ball_color,
            target_point=target_point,
        ),
        "Pickup",
        retry_frames=retry_frames,
    )
