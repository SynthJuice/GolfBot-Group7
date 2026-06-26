import math

from settings import (
    MAP_HEIGHT,
    MAP_WIDTH,
    RED_CROSS_AVOIDANCE_ENABLED,
    RED_CROSS_OBSTACLE_ARM_RATIO,
    RED_CROSS_OBSTACLE_MARGIN,
    RED_CROSS_OBSTACLE_MIN_ARM_WIDTH,
)


RED_CROSS_BLOCKED_VALUE = "X"


def create_empty_path_matrix(row_count=MAP_HEIGHT, col_count=MAP_WIDTH, value="."):
    row_count = max(1, int(row_count))
    col_count = max(1, int(col_count))
    return [[value for _col in range(col_count)] for _row in range(row_count)]


def clone_path_matrix(path_matrix):
    return [list(row) for row in path_matrix]


def clear_path_endpoint(path_matrix, point, radius=5, value="."):
    if point is None or not path_matrix:
        return

    row_count = len(path_matrix)
    col_count = len(path_matrix[0])
    row, col = point
    row = int(round(row))
    col = int(round(col))
    radius = max(0, int(round(radius)))

    for current_row in range(max(0, row - radius), min(row_count, row + radius + 1)):
        for current_col in range(max(0, col - radius), min(col_count, col + radius + 1)):
            path_matrix[current_row][current_col] = value


def point_in_obstacle_regions(point, regions):
    if point is None or not regions:
        return False

    row, col = point
    row = float(row)
    col = float(col)

    for top, left, bottom, right in regions:
        if top <= row <= bottom and left <= col <= right:
            return True

    return False


def _point_distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def choose_safe_path_lookahead(
    robot_path,
    start_point,
    regions,
    min_distance,
    max_distance,
    acceptance_radius=0.0,
):
    if not robot_path or len(robot_path) < 2:
        return None

    min_distance = max(0.0, float(min_distance))
    max_distance = max(min_distance, float(max_distance))
    acceptance_radius = max(0.0, float(acceptance_radius))
    travelled = 0.0
    previous_point = robot_path[0]
    best_before_min = None
    best_after_min = None

    for point in robot_path[1:]:
        travelled += _point_distance(previous_point, point)
        previous_point = point

        if travelled <= acceptance_radius:
            continue

        if _point_distance(start_point, point) <= acceptance_radius:
            continue

        if point_in_obstacle_regions(point, regions):
            continue

        if segment_intersects_regions(start_point, point, regions):
            continue

        if travelled < min_distance:
            best_before_min = point
            continue

        if travelled <= max_distance:
            best_after_min = point
            continue

        if best_after_min is not None:
            break

        return point

    if best_after_min is not None:
        return best_after_min

    return best_before_min


def clear_path_endpoint_preserving_obstacles(
    path_matrix,
    original_matrix,
    point,
    radius=5,
    value=".",
    allowed_original_values=(".", "W", "O"),
    blocked_regions=None,
):
    if point is None or not path_matrix or not original_matrix:
        return

    row_count = len(path_matrix)
    col_count = len(path_matrix[0])
    row, col = point
    row = int(round(row))
    col = int(round(col))
    radius = max(0, int(round(radius)))

    for current_row in range(max(0, row - radius), min(row_count, row + radius + 1)):
        for current_col in range(max(0, col - radius), min(col_count, col + radius + 1)):
            if point_in_obstacle_regions((current_row, current_col), blocked_regions):
                continue
            if original_matrix[current_row][current_col] in allowed_original_values:
                path_matrix[current_row][current_col] = value


def _bbox_center_matches_point(detection, tolerance=5.0):
    x1, y1, x2, y2 = detection.bbox
    center_row = (float(y1) + float(y2)) / 2.0
    center_col = (float(x1) + float(x2)) / 2.0
    point_row, point_col = detection.point

    return (
        abs(center_row - float(point_row)) <= tolerance
        and abs(center_col - float(point_col)) <= tolerance
    )


def _clamp_region(top, left, bottom, right, row_count, col_count):
    top = max(0, min(row_count - 1, int(top)))
    left = max(0, min(col_count - 1, int(left)))
    bottom = max(0, min(row_count - 1, int(bottom)))
    right = max(0, min(col_count - 1, int(right)))

    if bottom < top or right < left:
        return None

    return top, left, bottom, right


def _region_from_floats(top, left, bottom, right, row_count, col_count):
    return _clamp_region(
        math.floor(top),
        math.floor(left),
        math.ceil(bottom),
        math.ceil(right),
        row_count,
        col_count,
    )


def _red_cross_regions_from_bbox(detection, row_count, col_count, margin):
    x1, y1, x2, y2 = detection.bbox
    top = min(float(y1), float(y2))
    bottom = max(float(y1), float(y2))
    left = min(float(x1), float(x2))
    right = max(float(x1), float(x2))
    center_row = (top + bottom) / 2.0
    center_col = (left + right) / 2.0
    width = max(1.0, right - left)
    height = max(1.0, bottom - top)
    arm_ratio = max(0.05, min(1.0, float(RED_CROSS_OBSTACLE_ARM_RATIO)))
    arm_width = max(
        float(RED_CROSS_OBSTACLE_MIN_ARM_WIDTH),
        min(width, height) * arm_ratio,
    )
    half_arm_width = arm_width / 2.0

    vertical_arm = _region_from_floats(
        top - margin,
        center_col - half_arm_width - margin,
        bottom + margin,
        center_col + half_arm_width + margin,
        row_count,
        col_count,
    )
    horizontal_arm = _region_from_floats(
        center_row - half_arm_width - margin,
        left - margin,
        center_row + half_arm_width + margin,
        right + margin,
        row_count,
        col_count,
    )

    return [
        region for region in (vertical_arm, horizontal_arm)
        if region is not None
    ]


def _red_cross_regions_from_point(detection, row_count, col_count, margin):
    point_row, point_col = detection.point
    half_length = max(float(margin), float(RED_CROSS_OBSTACLE_MIN_ARM_WIDTH))
    half_arm_width = max(1.0, float(RED_CROSS_OBSTACLE_MIN_ARM_WIDTH) / 2.0)

    vertical_arm = _region_from_floats(
        float(point_row) - half_length - margin,
        float(point_col) - half_arm_width - margin,
        float(point_row) + half_length + margin,
        float(point_col) + half_arm_width + margin,
        row_count,
        col_count,
    )
    horizontal_arm = _region_from_floats(
        float(point_row) - half_arm_width - margin,
        float(point_col) - half_length - margin,
        float(point_row) + half_arm_width + margin,
        float(point_col) + half_length + margin,
        row_count,
        col_count,
    )

    return [
        region for region in (vertical_arm, horizontal_arm)
        if region is not None
    ]


def _red_cross_regions(detection, row_count, col_count, margin):
    margin = max(0, int(round(margin)))

    if _bbox_center_matches_point(detection):
        return _red_cross_regions_from_bbox(detection, row_count, col_count, margin)

    return _red_cross_regions_from_point(detection, row_count, col_count, margin)


def red_cross_obstacle_regions(
    path_matrix,
    vision_scene,
    margin=RED_CROSS_OBSTACLE_MARGIN,
):
    if (
        not RED_CROSS_AVOIDANCE_ENABLED
        or vision_scene is None
    ):
        return []

    if path_matrix is not None and path_matrix:
        row_count = len(path_matrix)
        col_count = len(path_matrix[0])
    else:
        row_count = MAP_HEIGHT
        col_count = MAP_WIDTH

    regions = []

    for detection in vision_scene.detections_for("redcross"):
        regions.extend(_red_cross_regions(detection, row_count, col_count, margin))

    return regions


def segment_intersects_regions(start_point, end_point, regions, step_size=3.0):
    if not regions:
        return False

    start_row, start_col = start_point
    end_row, end_col = end_point
    delta_row = float(end_row) - float(start_row)
    delta_col = float(end_col) - float(start_col)
    distance = math.hypot(delta_row, delta_col)

    if distance <= 0.001:
        sample_count = 1
    else:
        sample_count = max(1, int(math.ceil(distance / max(1.0, float(step_size)))))

    for sample_index in range(sample_count + 1):
        t = float(sample_index) / float(sample_count)
        row = float(start_row) + delta_row * t
        col = float(start_col) + delta_col * t

        for top, left, bottom, right in regions:
            if top <= row <= bottom and left <= col <= right:
                return True

    return False


def mark_red_cross_obstacles(
    path_matrix,
    vision_scene,
    margin=RED_CROSS_OBSTACLE_MARGIN,
    value=RED_CROSS_BLOCKED_VALUE,
):
    regions = red_cross_obstacle_regions(path_matrix, vision_scene, margin=margin)

    for region in regions:
        top, left, bottom, right = region

        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                path_matrix[row][col] = value

    if regions:
        print(
            "Red cross avoidance: marked {} obstacle arm(s) with margin {} map units".format(
                len(regions),
                int(round(margin)),
            )
        )

    return len(regions)
