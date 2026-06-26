import math

from settings import MAP_HEIGHT, MAP_WIDTH


def path_is_valid(robot_path):
    return robot_path and not isinstance(robot_path, str)


def point_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def map_point_is_valid(point):
    row, col = point
    return 0 <= row < MAP_HEIGHT and 0 <= col < MAP_WIDTH


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def clamp_map_point(point, margin=2):
    row, col = point
    return (
        int(round(clamp(row, margin, MAP_HEIGHT - 1 - margin))),
        int(round(clamp(col, margin, MAP_WIDTH - 1 - margin))),
    )


def heading_from_map_points(start, end):
    start_row, start_col = start
    end_row, end_col = end
    delta_row = float(end_row - start_row)
    delta_col = float(end_col - start_col)

    if delta_row == 0.0 and delta_col == 0.0:
        return 0.0

    return math.degrees(math.atan2(delta_row, delta_col)) % 360.0


def robot_center_point(robot_pose):
    center_x, center_y, _heading = robot_pose
    return int(round(center_y)), int(round(center_x))


def sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def simplify_path_for_robot(robot_path, min_spacing=40):
    if not path_is_valid(robot_path):
        return robot_path

    if len(robot_path) <= 2:
        return robot_path

    min_spacing = max(1.0, float(min_spacing))
    waypoints = [robot_path[0]]
    previous_direction = None

    for index in range(1, len(robot_path)):
        previous_point = robot_path[index - 1]
        current_point = robot_path[index]
        current_direction = (
            sign(current_point[0] - previous_point[0]),
            sign(current_point[1] - previous_point[1]),
        )

        direction_changed = (
            previous_direction is not None
            and current_direction != previous_direction
        )
        far_enough = point_distance(waypoints[-1], current_point) >= min_spacing

        if direction_changed and point_distance(waypoints[-1], previous_point) >= 8.0:
            if waypoints[-1] != previous_point:
                waypoints.append(previous_point)
        elif far_enough:
            waypoints.append(current_point)

        previous_direction = current_direction

    if waypoints[-1] != robot_path[-1]:
        waypoints.append(robot_path[-1])

    cleaned = [waypoints[0]]
    for point in waypoints[1:]:
        if point == robot_path[-1] or point_distance(cleaned[-1], point) >= 6.0:
            cleaned.append(point)

    print(
        "Simplified path: raw_points={}, waypoints={}".format(
            len(robot_path),
            len(cleaned),
        )
    )
    return cleaned
