import math
import os
from dataclasses import dataclass

from settings import (
    DELIVERY_CLAW_TO_MARKER_DISTANCE,
    DELIVERY_GOAL_A_MARKER_FALLBACK,
    DELIVERY_GOAL_B_MARKER_FALLBACK,
    DELIVERY_USE_FIXED_GOALS,
    MAP_HEIGHT,
    MAP_WIDTH,
    VISION_CONFIDENCE,
    VISION_DEBUG,
    VISION_DETECTION_ENABLED,
    VISION_IOU,
    VISION_LIVE_VIEW_ENABLED,
    VISION_LIVE_VIEW_MAX_WIDTH,
    VISION_LIVE_VIEW_WINDOW_NAME,
    VISION_MODEL_IMAGE_SIZE,
    VISION_MODEL_PATH,
)


_MODEL = None
_LOAD_ATTEMPTED = False
_UNAVAILABLE_MESSAGE_PRINTED = False
_LIVE_VIEW_UNAVAILABLE_MESSAGE_PRINTED = False
_LIVE_VIEW_FAILED = False
_LIVE_VIEW_QUIT_REQUESTED = False
_PATH_OVERLAYS = []

_CLASS_KIND_BY_NAME = {
    "biggoal": "goal",
    "smallgoal": "goal",
    "goal": "goal",
    "claw": "claw",
    "openclaw": "claw",
    "orangeball": "orangeball",
    "orangebal": "orangeball",
    "orange": "orangeball",
    "redcross": "redcross",
    "robot": "robot",
    "whiteball": "whiteball",
    "whitebal": "whiteball",
    "white": "whiteball",
}

_KIND_COLORS = {
    "claw": (0, 255, 255),
    "goal": (255, 0, 255),
    "orangeball": (0, 140, 255),
    "redcross": (0, 0, 255),
    "robot": (255, 180, 0),
    "whiteball": (255, 255, 255),
}

_LABEL_HIDDEN_KINDS = {"goal", "whiteball"}


@dataclass(frozen=True)
class VisionDetection:
    class_name: str
    kind: str
    confidence: float
    bbox: tuple
    point: tuple


@dataclass
class VisionScene:
    detections: list

    def detections_for(self, kind):
        return [detection for detection in self.detections if detection.kind == kind]

    def detections_for_class(self, class_name):
        normalized_name = _normalize_class_name(class_name)
        return [
            detection for detection in self.detections
            if _normalize_class_name(detection.class_name) == normalized_name
        ]

    def best(self, kind):
        candidates = self.detections_for(kind)

        if not candidates:
            return None

        return max(candidates, key=lambda detection: detection.confidence)

    def open_claw_detection(self):
        candidates = self.detections_for_class("openclaw")

        if not candidates:
            return None

        return max(candidates, key=lambda detection: detection.confidence)

    def open_claw_visible(self):
        return self.open_claw_detection() is not None

    def _bbox_matches_map_coordinates(self, detection, tolerance=3.0):
        x1, y1, x2, y2 = detection.bbox
        bbox_center_row = (float(y1) + float(y2)) / 2.0
        bbox_center_col = (float(x1) + float(x2)) / 2.0
        point_row, point_col = detection.point

        return (
            abs(bbox_center_row - float(point_row)) <= tolerance
            and abs(bbox_center_col - float(point_col)) <= tolerance
        )

    def _goal_opening_point(self, detection, side):
        if not self._bbox_matches_map_coordinates(detection):
            return detection.point

        x1, y1, x2, y2 = detection.bbox
        row = int(round((float(y1) + float(y2)) / 2.0))

        if side == "left":
            col = int(round(max(float(x1), float(x2))))
        elif side == "right":
            col = int(round(min(float(x1), float(x2))))
        else:
            raise ValueError("side must be 'left' or 'right'")

        return row, col

    def ball_points(self, color):
        kind = None

        if color == "W":
            kind = "whiteball"
        elif color == "O":
            kind = "orangeball"

        if kind is None:
            return []

        detections = sorted(
            self.detections_for(kind),
            key=lambda detection: detection.confidence,
            reverse=True,
        )
        return [detection.point for detection in detections]

    def grappler_point(self):
        detection = self.best("claw")

        if detection is None:
            return None

        return detection.point

    def robot_pose(self, fallback=None):
        robot = self.best("robot")

        if robot is None:
            return fallback

        robot_row, robot_col = robot.point
        claw = self.best("claw")

        if claw is None:
            if fallback is None:
                return None

            return robot_col, robot_row, fallback[2]

        claw_row, claw_col = claw.point
        delta_row = float(claw_row - robot_row)
        delta_col = float(claw_col - robot_col)

        if delta_row == 0.0 and delta_col == 0.0:
            if fallback is None:
                return None

            return robot_col, robot_row, fallback[2]

        heading = math.degrees(math.atan2(delta_row, delta_col)) % 360.0
        return robot_col, robot_row, heading

    def goal_marker(self, side):
        goals = self.detections_for("goal")

        if not goals:
            return None

        if side == "left":
            detection = min(goals, key=lambda detection: detection.point[1])
            return self._goal_opening_point(detection, side)

        if side == "right":
            detection = max(goals, key=lambda detection: detection.point[1])
            return self._goal_opening_point(detection, side)

        raise ValueError("side must be 'left' or 'right'")

    def goal_markers(self):
        goals = self.detections_for("goal")

        if len(goals) < 2:
            return None

        left = self.goal_marker("left")
        right = self.goal_marker("right")

        if left == right:
            return None

        return right, left

    def summary(self):
        if not self.detections:
            return "none"

        return ", ".join(
            "{}:{}@{}({:.2f})".format(
                detection.kind,
                detection.class_name,
                detection.point,
                detection.confidence,
            )
            for detection in self.detections
        )


def _normalize_class_name(class_name):
    return (
        str(class_name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def _print_unavailable_once(message):
    global _UNAVAILABLE_MESSAGE_PRINTED

    if not _UNAVAILABLE_MESSAGE_PRINTED:
        print(message)
        _UNAVAILABLE_MESSAGE_PRINTED = True


def _load_model():
    global _LOAD_ATTEMPTED
    global _MODEL

    if not VISION_DETECTION_ENABLED:
        return None

    if _MODEL is not None:
        return _MODEL

    if _LOAD_ATTEMPTED:
        return None

    _LOAD_ATTEMPTED = True

    if not os.path.exists(VISION_MODEL_PATH):
        _print_unavailable_once(
            "Vision detection disabled: model file not found at {}".format(
                VISION_MODEL_PATH
            )
        )
        return None

    try:
        from ultralytics import YOLO
    except ImportError:
        _print_unavailable_once(
            "Vision detection disabled: install ultralytics to use {}".format(
                VISION_MODEL_PATH
            )
        )
        return None

    _MODEL = YOLO(VISION_MODEL_PATH)
    print("Vision detection loaded model:", VISION_MODEL_PATH)
    print("Vision detection classes:", getattr(_MODEL, "names", {}))
    return _MODEL


def _tensor_value(value):
    if hasattr(value, "item"):
        return value.item()

    return value


def _tensor_list(values):
    if hasattr(values, "tolist"):
        return values.tolist()

    return list(values)


def _mapped_point(center_x, center_y, point_mapper):
    if point_mapper is None:
        return int(round(center_y)), int(round(center_x))

    return point_mapper(center_x, center_y)


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _scaled_xy(x, y, scale):
    return int(round(float(x) * scale)), int(round(float(y) * scale))


def _map_point_to_display_xy(point, frame_shape, scale):
    row, col = point
    frame_height, frame_width = frame_shape[:2]

    if MAP_WIDTH > 1 and frame_width > 1:
        x = float(col) * float(frame_width - 1) / float(MAP_WIDTH - 1)
    else:
        x = float(col)

    if MAP_HEIGHT > 1 and frame_height > 1:
        y = float(row) * float(frame_height - 1) / float(MAP_HEIGHT - 1)
    else:
        y = float(row)

    return _scaled_xy(x, y, scale)


def _draw_label(cv, image, text, x, y, color):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    text_width, text_height = cv.getTextSize(text, font, font_scale, thickness)[0]
    top = max(0, y - text_height - 8)
    left = max(0, x)
    right = min(image.shape[1] - 1, left + text_width + 6)
    bottom = min(image.shape[0] - 1, top + text_height + 6)

    cv.rectangle(image, (left, top), (right, bottom), color, -1)
    cv.putText(
        image,
        text,
        (left + 3, bottom - 4),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv.LINE_AA,
    )


def _is_map_point(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False

    try:
        float(value[0])
        float(value[1])
    except (TypeError, ValueError):
        return False

    return True


def _as_overlay_paths(paths, label, color):
    if paths is None:
        return []

    if isinstance(paths, dict):
        paths = [paths]
    elif paths and _is_map_point(paths[0]):
        paths = [{"points": paths, "label": label, "color": color}]

    overlays = []

    for index, path in enumerate(paths):
        if isinstance(path, dict):
            points = path.get("points", [])
            path_label = path.get("label", label)
            path_color = path.get("color", color)
        else:
            points = path
            path_label = label
            path_color = color

        clean_points = [
            (int(round(point[0])), int(round(point[1])))
            for point in points
            if _is_map_point(point)
        ]

        if len(clean_points) < 2:
            continue

        if not path_label and len(paths) > 1:
            path_label = "Path {}".format(index + 1)

        overlays.append(
            {
                "points": clean_points,
                "label": path_label,
                "color": path_color,
            }
        )

    return overlays


def set_vision_path_overlay(paths, label="Planned path", color=(0, 255, 255)):
    global _PATH_OVERLAYS
    _PATH_OVERLAYS = _as_overlay_paths(paths, label, color)


def has_vision_path_overlay():
    return bool(_PATH_OVERLAYS)


def clear_vision_path_overlay():
    global _PATH_OVERLAYS
    _PATH_OVERLAYS = []


def _draw_path_overlay(cv, image, frame_shape, scale):
    if not _PATH_OVERLAYS:
        return

    for overlay in _PATH_OVERLAYS:
        points = overlay["points"]
        color = overlay["color"]
        display_points = [
            _map_point_to_display_xy(point, frame_shape, scale)
            for point in points
        ]

        for index in range(1, len(display_points)):
            cv.line(
                image,
                display_points[index - 1],
                display_points[index],
                color,
                2,
                cv.LINE_AA,
            )

        for point in display_points:
            cv.circle(image, point, 2, color, -1)

        start_x, start_y = display_points[0]
        end_x, end_y = display_points[-1]
        cv.circle(image, (start_x, start_y), 6, (0, 255, 0), -1)
        cv.circle(image, (end_x, end_y), 7, (0, 0, 255), -1)

        label = overlay.get("label")
        if label:
            _draw_label(cv, image, label, start_x + 8, start_y - 8, color)


def _fixed_goal_claw_target(goal_name, marker):
    row, col = marker
    distance = float(DELIVERY_CLAW_TO_MARKER_DISTANCE)

    if goal_name == "A":
        return row, int(round(float(col) - distance))

    return row, int(round(float(col) + distance))


def _draw_goal_marker_overlay(cv, image, frame_shape, scale, goal_name, marker, color):
    marker_x, marker_y = _map_point_to_display_xy(marker, frame_shape, scale)
    target = _fixed_goal_claw_target(goal_name, marker)
    target_x, target_y = _map_point_to_display_xy(target, frame_shape, scale)
    tick = 22

    cv.line(image, (marker_x, marker_y - tick), (marker_x, marker_y + tick), color, 3)
    cv.line(image, (marker_x - tick, marker_y), (marker_x + tick, marker_y), color, 2)
    cv.circle(image, (marker_x, marker_y), 7, color, 2)
    cv.circle(image, (target_x, target_y), 5, (0, 255, 0), -1)
    cv.line(image, (target_x, target_y), (marker_x, marker_y), (0, 255, 0), 1)


def _draw_fixed_goal_overlay(cv, image, frame_shape, scale):
    if not DELIVERY_USE_FIXED_GOALS:
        return

    _draw_goal_marker_overlay(
        cv,
        image,
        frame_shape,
        scale,
        "A",
        DELIVERY_GOAL_A_MARKER_FALLBACK,
        (255, 255, 0),
    )
    _draw_goal_marker_overlay(
        cv,
        image,
        frame_shape,
        scale,
        "B",
        DELIVERY_GOAL_B_MARKER_FALLBACK,
        (255, 255, 0),
    )


def _live_view_frame(cv, frame):
    if VISION_LIVE_VIEW_MAX_WIDTH <= 0:
        return frame.copy(), 1.0

    height, width = frame.shape[:2]
    scale = float(VISION_LIVE_VIEW_MAX_WIDTH) / float(width)

    if abs(scale - 1.0) < 0.01:
        return frame.copy(), 1.0

    resized = cv.resize(
        frame,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR,
    )
    return resized, scale


def show_vision_live_view(frame, scene):
    global _LIVE_VIEW_FAILED
    global _LIVE_VIEW_QUIT_REQUESTED
    global _LIVE_VIEW_UNAVAILABLE_MESSAGE_PRINTED

    if not VISION_LIVE_VIEW_ENABLED or _LIVE_VIEW_FAILED:
        return

    try:
        import cv2 as cv
    except ImportError:
        if not _LIVE_VIEW_UNAVAILABLE_MESSAGE_PRINTED:
            print("Vision live view disabled: install opencv-python to show model detections")
            _LIVE_VIEW_UNAVAILABLE_MESSAGE_PRINTED = True
        return

    try:
        display_frame, scale = _live_view_frame(cv, frame)
        _draw_fixed_goal_overlay(cv, display_frame, frame.shape, scale)

        for detection in scene.detections:
            color = _KIND_COLORS.get(detection.kind, (0, 255, 0))
            x1, y1, x2, y2 = detection.bbox
            left, top = _scaled_xy(x1, y1, scale)
            right, bottom = _scaled_xy(x2, y2, scale)
            center_x, center_y = _bbox_center(detection.bbox)
            center = _scaled_xy(center_x, center_y, scale)
            row, col = detection.point
            label = "{} {:.2f} map=({}, {})".format(
                detection.class_name,
                detection.confidence,
                row,
                col,
            )

            cv.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            cv.circle(display_frame, center, 4, color, -1)
            if detection.kind not in _LABEL_HIDDEN_KINDS:
                _draw_label(cv, display_frame, label, left, top, color)

        robot = scene.best("robot")
        claw = scene.best("claw")

        if robot is not None and claw is not None:
            robot_center = _scaled_xy(*_bbox_center(robot.bbox), scale)
            claw_center = _scaled_xy(*_bbox_center(claw.bbox), scale)
            cv.line(display_frame, robot_center, claw_center, (0, 255, 0), 2)

        _draw_path_overlay(cv, display_frame, frame.shape, scale)

        cv.putText(
            display_frame,
            "Model detections: {} | q quits".format(len(scene.detections)),
            (10, 24),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )
        cv.imshow(VISION_LIVE_VIEW_WINDOW_NAME, display_frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            _LIVE_VIEW_QUIT_REQUESTED = True
    except Exception as exc:
        if not _LIVE_VIEW_UNAVAILABLE_MESSAGE_PRINTED:
            print("Vision live view disabled: {}".format(exc))
            _LIVE_VIEW_UNAVAILABLE_MESSAGE_PRINTED = True
        _LIVE_VIEW_FAILED = True


def vision_live_view_quit_requested():
    return _LIVE_VIEW_QUIT_REQUESTED


def detect_vision_scene(frame, point_mapper=None):
    model = _load_model()

    if model is None:
        return None

    results = model.predict(
        source=frame,
        conf=VISION_CONFIDENCE,
        iou=VISION_IOU,
        imgsz=VISION_MODEL_IMAGE_SIZE,
        verbose=False,
    )

    if not results:
        scene = VisionScene([])
        show_vision_live_view(frame, scene)
        return scene

    result = results[0]
    names = getattr(result, "names", getattr(model, "names", {}))
    detections = []

    for box in getattr(result, "boxes", []):
        class_id = int(_tensor_value(box.cls[0]))
        class_name = names[class_id] if isinstance(names, dict) else names[class_id]
        kind = _CLASS_KIND_BY_NAME.get(_normalize_class_name(class_name))

        if kind is None:
            continue

        confidence = float(_tensor_value(box.conf[0]))
        x1, y1, x2, y2 = [float(value) for value in _tensor_list(box.xyxy[0])]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        point = _mapped_point(center_x, center_y, point_mapper)

        detections.append(
            VisionDetection(
                class_name=class_name,
                kind=kind,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                point=point,
            )
        )

    scene = VisionScene(detections)

    if VISION_DEBUG:
        print("Vision detections:", scene.summary())

    show_vision_live_view(frame, scene)
    return scene
