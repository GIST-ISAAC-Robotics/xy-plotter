import cv2
import numpy as np
import math
import os
import tkinter as tk
from tkinter import filedialog


# =========================
# Paper / drawing settings
# =========================

# A4 landscape
PAPER_WIDTH_MM = 297.0
PAPER_HEIGHT_MM = 210.0

MARGIN_LEFT_MM = 10.0
MARGIN_RIGHT_MM = 10.0
MARGIN_BOTTOM_MM = 10.0
MARGIN_TOP_MM = 10.0

DRAWABLE_WIDTH_MM = PAPER_WIDTH_MM - MARGIN_LEFT_MM - MARGIN_RIGHT_MM
DRAWABLE_HEIGHT_MM = PAPER_HEIGHT_MM - MARGIN_BOTTOM_MM - MARGIN_TOP_MM

SAFE_MIN_X = 0.0
SAFE_MAX_X = PAPER_WIDTH_MM
SAFE_MIN_Y = 0.0
SAFE_MAX_Y = PAPER_HEIGHT_MM

MIN_SAFE_SEGMENT_LENGTH_MM = 0.05


# =========================
# Plotter / G-code settings
# =========================

PEN_DOWN_CMD = "M05 S255"
PEN_UP_CMD = "M03 S40"

DEFAULT_TRAVEL_FEED = 3800
DEFAULT_DRAW_FEED = 3100

MIN_FEED_RATE = 100
MAX_FEED_RATE = 8000

DWELL_AFTER_PEN = 0.08

WINDOW_NAME = "Canny to G-code GUI"


# =========================
# Default GUI values
# =========================

DEFAULT_MIN_LENGTH = 30
DEFAULT_SIMPLIFY_X10 = 15
DEFAULT_CLOSE = 0
DEFAULT_BLUR_RAW = 2
DEFAULT_SIZE_PERCENT = 100
DEFAULT_ROTATE_INDEX = 0

DENSITY_TARGET = 0.035


def select_image_file():
    root = tk.Tk()
    root.withdraw()

    path = filedialog.askopenfilename(
        title="Select image",
        filetypes=[
            ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return path


def nothing(x):
    pass


# =========================
# Image rotation
# =========================

def rotate_image(gray, rotate_index):
    rotate_index = rotate_index % 4

    if rotate_index == 0:
        return gray.copy()
    elif rotate_index == 1:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    elif rotate_index == 2:
        return cv2.rotate(gray, cv2.ROTATE_180)
    else:
        return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)


def rotation_label(rotate_index):
    rotate_index = rotate_index % 4

    if rotate_index == 0:
        return "0 deg"
    elif rotate_index == 1:
        return "90 deg CW"
    elif rotate_index == 2:
        return "180 deg"
    else:
        return "90 deg CCW"


# =========================
# Path utilities
# =========================

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def path_length_mm(path):
    total = 0.0

    for i in range(1, len(path)):
        total += dist(path[i - 1], path[i])

    return total


def order_paths(paths):
    ordered = []
    current = (0.0, 0.0)
    remain = paths[:]

    while remain:
        best_i = 0
        best_reverse = False
        best_dist = float("inf")

        for i, path in enumerate(remain):
            d_start = dist(current, path[0])
            d_end = dist(current, path[-1])

            if d_start < best_dist:
                best_dist = d_start
                best_i = i
                best_reverse = False

            if d_end < best_dist:
                best_dist = d_end
                best_i = i
                best_reverse = True

        path = remain.pop(best_i)

        if best_reverse:
            path = path[::-1]

        ordered.append(path)
        current = path[-1]

    return ordered


# =========================
# Auto Canny
# =========================

def auto_canny_thresholds_by_gradient(gray, blur_size=5):
    if blur_size > 1:
        work = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    else:
        work = gray.copy()

    grad_x = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(grad_x, grad_y)

    if magnitude.max() <= 0:
        return 50, 150

    magnitude = np.uint8(np.clip(magnitude / magnitude.max() * 255, 0, 255))

    nonzero = magnitude[magnitude > 0]
    if len(nonzero) == 0:
        return 50, 150

    otsu_threshold, _ = cv2.threshold(
        magnitude,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    high = int(otsu_threshold)
    low = int(high * 0.4)

    high = max(30, min(high, 500))
    low = max(10, min(low, high - 1))

    return low, high


def auto_canny_thresholds_by_edge_density(gray, blur_size=5, target_density=0.035):
    if blur_size > 1:
        work = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    else:
        work = gray.copy()

    best_low = 80
    best_high = 160
    best_error = float("inf")

    for high in range(30, 401, 5):
        low = int(high * 0.4)

        edges = cv2.Canny(work, low, high)
        density = np.count_nonzero(edges) / edges.size

        error = abs(density - target_density)

        if error < best_error:
            best_error = error
            best_low = low
            best_high = high

    return best_low, best_high


# =========================
# GUI values
# =========================

def get_trackbar_values():
    low = cv2.getTrackbarPos("Canny low", WINDOW_NAME)
    high = cv2.getTrackbarPos("Canny high", WINDOW_NAME)
    blur_raw = cv2.getTrackbarPos("Blur", WINDOW_NAME)
    min_len = cv2.getTrackbarPos("Min length", WINDOW_NAME)
    epsilon_raw = cv2.getTrackbarPos("Simplify x10", WINDOW_NAME)
    close_iter = cv2.getTrackbarPos("Close", WINDOW_NAME)
    size_percent = cv2.getTrackbarPos("Size %", WINDOW_NAME)
    rotate_index = cv2.getTrackbarPos("Rotate 90", WINDOW_NAME)
    travel_feed = cv2.getTrackbarPos("Travel F", WINDOW_NAME)
    draw_feed = cv2.getTrackbarPos("Draw F", WINDOW_NAME)

    if high <= low:
        high = low + 1

    if size_percent < 5:
        size_percent = 5

    if travel_feed < MIN_FEED_RATE:
        travel_feed = MIN_FEED_RATE

    if draw_feed < MIN_FEED_RATE:
        draw_feed = MIN_FEED_RATE

    blur_size = blur_raw * 2 + 1
    epsilon = epsilon_raw / 10.0

    return (
        low,
        high,
        blur_size,
        min_len,
        epsilon,
        close_iter,
        size_percent,
        rotate_index,
        travel_feed,
        draw_feed
    )


# =========================
# Image processing
# =========================

def make_edges(gray, low, high, blur_size, close_iter):
    if blur_size > 1:
        work = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    else:
        work = gray.copy()

    edges = cv2.Canny(work, low, high)

    if close_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    return edges


def extract_paths(edges, min_len, epsilon):
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    paths_px = []

    for contour in contours:
        length = cv2.arcLength(contour, False)

        if length < min_len:
            continue

        if epsilon > 0:
            approx = cv2.approxPolyDP(contour, epsilon, False)
        else:
            approx = contour

        path = []

        for point in approx:
            x, y = point[0]
            path.append((float(x), float(y)))

        if len(path) >= 2:
            paths_px.append(path)

    return paths_px


# =========================
# Coordinate conversion
# =========================

def convert_paths_to_mm(paths_px, image_width, image_height, size_percent):
    fit_scale = min(DRAWABLE_WIDTH_MM / image_width, DRAWABLE_HEIGHT_MM / image_height)
    scale = fit_scale * (size_percent / 100.0)

    drawing_width_mm = image_width * scale
    drawing_height_mm = image_height * scale

    x_offset = (PAPER_WIDTH_MM - drawing_width_mm) / 2.0
    y_offset = (PAPER_HEIGHT_MM - drawing_height_mm) / 2.0

    paths_mm = []

    for path in paths_px:
        mm_path = []

        for x, y in path:
            mm_x = x * scale + x_offset
            mm_y = (image_height - y) * scale + y_offset
            mm_path.append((mm_x, mm_y))

        paths_mm.append(mm_path)

    return paths_mm, scale, drawing_width_mm, drawing_height_mm


# =========================
# Safety clipping
# =========================

INSIDE = 0
LEFT = 1
RIGHT = 2
BOTTOM = 4
TOP = 8


def compute_out_code(x, y, xmin, xmax, ymin, ymax):
    code = INSIDE

    if x < xmin:
        code |= LEFT
    elif x > xmax:
        code |= RIGHT

    if y < ymin:
        code |= BOTTOM
    elif y > ymax:
        code |= TOP

    return code


def clip_line_to_rect(p1, p2, xmin, xmax, ymin, ymax):
    x1, y1 = p1
    x2, y2 = p2

    out1 = compute_out_code(x1, y1, xmin, xmax, ymin, ymax)
    out2 = compute_out_code(x2, y2, xmin, xmax, ymin, ymax)

    clipped = False

    while True:
        if not (out1 | out2):
            return True, (x1, y1), (x2, y2), clipped

        if out1 & out2:
            return False, None, None, clipped

        out = out1 if out1 else out2

        if out & TOP:
            if y2 == y1:
                return False, None, None, clipped
            x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
            y = ymax

        elif out & BOTTOM:
            if y2 == y1:
                return False, None, None, clipped
            x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
            y = ymin

        elif out & RIGHT:
            if x2 == x1:
                return False, None, None, clipped
            y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
            x = xmax

        else:
            if x2 == x1:
                return False, None, None, clipped
            y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
            x = xmin

        clipped = True

        if out == out1:
            x1, y1 = x, y
            out1 = compute_out_code(x1, y1, xmin, xmax, ymin, ymax)
        else:
            x2, y2 = x, y
            out2 = compute_out_code(x2, y2, xmin, xmax, ymin, ymax)


def same_point(a, b, eps=1e-6):
    return abs(a[0] - b[0]) < eps and abs(a[1] - b[1]) < eps


def sanitize_paths_to_paper(paths_mm):
    safe_paths = []

    stats = {
        "input_paths": len(paths_mm),
        "input_segments": 0,
        "output_paths": 0,
        "output_segments": 0,
        "clipped_segments": 0,
        "dropped_segments": 0,
        "out_of_bounds_points": 0
    }

    for path in paths_mm:
        if len(path) < 2:
            continue

        current_path = []

        for i in range(1, len(path)):
            p1 = path[i - 1]
            p2 = path[i]

            stats["input_segments"] += 1

            for p in (p1, p2):
                if (
                    p[0] < SAFE_MIN_X or p[0] > SAFE_MAX_X or
                    p[1] < SAFE_MIN_Y or p[1] > SAFE_MAX_Y
                ):
                    stats["out_of_bounds_points"] += 1

            ok, c1, c2, clipped = clip_line_to_rect(
                p1,
                p2,
                SAFE_MIN_X,
                SAFE_MAX_X,
                SAFE_MIN_Y,
                SAFE_MAX_Y
            )

            if not ok:
                stats["dropped_segments"] += 1

                if len(current_path) >= 2 and path_length_mm(current_path) >= MIN_SAFE_SEGMENT_LENGTH_MM:
                    safe_paths.append(current_path)

                current_path = []
                continue

            if dist(c1, c2) < MIN_SAFE_SEGMENT_LENGTH_MM:
                stats["dropped_segments"] += 1
                continue

            if clipped:
                stats["clipped_segments"] += 1

            if not current_path:
                current_path = [c1, c2]
            else:
                if same_point(current_path[-1], c1):
                    current_path.append(c2)
                else:
                    if len(current_path) >= 2 and path_length_mm(current_path) >= MIN_SAFE_SEGMENT_LENGTH_MM:
                        safe_paths.append(current_path)

                    current_path = [c1, c2]

            stats["output_segments"] += 1

        if len(current_path) >= 2 and path_length_mm(current_path) >= MIN_SAFE_SEGMENT_LENGTH_MM:
            safe_paths.append(current_path)

    stats["output_paths"] = len(safe_paths)

    return safe_paths, stats


def validate_paths_in_paper(paths_mm):
    for path in paths_mm:
        for x, y in path:
            if x < SAFE_MIN_X - 1e-6 or x > SAFE_MAX_X + 1e-6:
                return False
            if y < SAFE_MIN_Y - 1e-6 or y > SAFE_MAX_Y + 1e-6:
                return False

    return True


# =========================
# G-code output
# =========================

def save_gcode(paths_mm, output_path, travel_feed, draw_feed):
    if not validate_paths_in_paper(paths_mm):
        raise ValueError("Unsafe path detected: some coordinates are outside the A4 paper range.")

    ordered_paths = order_paths(paths_mm)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("G21\n")
        f.write("G90\n")
        f.write("G92 X0 Y0\n")

        f.write(PEN_UP_CMD + "\n")
        f.write(f"G4 P{DWELL_AFTER_PEN}\n")
        f.write(f"G1 F{travel_feed}\n")

        for path in ordered_paths:
            if len(path) < 2:
                continue

            start_x, start_y = path[0]

            f.write(f"G1 X{start_x:.3f} Y{start_y:.3f}\n")
            f.write("G4 P0\n")
            f.write(PEN_DOWN_CMD + "\n")
            f.write(f"G4 P{DWELL_AFTER_PEN}\n")
            f.write(f"G1 F{draw_feed}\n")

            for x, y in path[1:]:
                if x < SAFE_MIN_X or x > SAFE_MAX_X or y < SAFE_MIN_Y or y > SAFE_MAX_Y:
                    raise ValueError(f"Unsafe coordinate blocked: X{x:.3f} Y{y:.3f}")

                f.write(f"G1 X{x:.3f} Y{y:.3f}\n")

            f.write("G4 P0\n")
            f.write(PEN_UP_CMD + "\n")
            f.write(f"G4 P{DWELL_AFTER_PEN}\n")
            f.write(f"G1 F{travel_feed}\n")

        f.write(PEN_UP_CMD + "\n")
        f.write(f"G4 P{DWELL_AFTER_PEN}\n")
        f.write(f"G1 X0 Y0 F{travel_feed}\n")
        f.write("M2\n")

    return os.path.abspath(output_path)


# =========================
# Preview helpers
# =========================

def resize_to_fit(img, target_w, target_h):
    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)

    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def add_title(panel, title):
    result = panel.copy()

    cv2.rectangle(result, (0, 0), (result.shape[1], 38), (30, 30, 30), -1)
    cv2.putText(
        result,
        title,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2
    )

    return result


def make_original_panel(gray, panel_w, panel_h, rotate_index):
    original = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    panel = resize_to_fit(original, panel_w, panel_h)
    return add_title(panel, "Input image - " + rotation_label(rotate_index))


def make_edge_panel(edges, panel_w, panel_h):
    edge_bgr = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    edge_bgr[:, :, 1] = edges

    panel = resize_to_fit(edge_bgr, panel_w, panel_h)
    return add_title(panel, "Canny edges")


def mm_to_preview_px(x_mm, y_mm, scale_px_per_mm, canvas_h):
    px = int(round(x_mm * scale_px_per_mm))
    py = int(round(canvas_h - y_mm * scale_px_per_mm))
    return px, py


def make_gcode_page_panel(paths_mm, panel_w, panel_h):
    scale_px_per_mm = 3.0
    canvas_w = int(round(PAPER_WIDTH_MM * scale_px_per_mm))
    canvas_h = int(round(PAPER_HEIGHT_MM * scale_px_per_mm))

    page_img = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    cv2.rectangle(page_img, (0, 0), (canvas_w - 1, canvas_h - 1), (0, 0, 0), 2)

    left = int(round(MARGIN_LEFT_MM * scale_px_per_mm))
    right = int(round((PAPER_WIDTH_MM - MARGIN_RIGHT_MM) * scale_px_per_mm))
    top = int(round(MARGIN_TOP_MM * scale_px_per_mm))
    bottom = int(round((PAPER_HEIGHT_MM - MARGIN_BOTTOM_MM) * scale_px_per_mm))

    rect_y1 = canvas_h - bottom
    rect_y2 = canvas_h - top
    cv2.rectangle(page_img, (left, rect_y1), (right, rect_y2), (220, 220, 220), 1)

    origin_px = mm_to_preview_px(0, 0, scale_px_per_mm, canvas_h)
    cv2.circle(page_img, origin_px, 4, (0, 0, 255), -1)
    cv2.putText(page_img, "(0,0)", (8, canvas_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    ordered_paths = order_paths(paths_mm)

    current = None

    for path in ordered_paths:
        if len(path) < 2:
            continue

        start = path[0]
        end = path[-1]

        if current is not None:
            p1 = mm_to_preview_px(current[0], current[1], scale_px_per_mm, canvas_h)
            p2 = mm_to_preview_px(start[0], start[1], scale_px_per_mm, canvas_h)
            cv2.line(page_img, p1, p2, (200, 200, 200), 1)

        pts = []
        for x, y in path:
            pts.append(mm_to_preview_px(x, y, scale_px_per_mm, canvas_h))

        pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(page_img, [pts], False, (0, 0, 0), 1)

        start_px = mm_to_preview_px(start[0], start[1], scale_px_per_mm, canvas_h)
        cv2.circle(page_img, start_px, 2, (0, 0, 255), -1)

        current = end

    panel = resize_to_fit(page_img, panel_w, panel_h)
    return add_title(panel, "Safe G-code path on A4")


def make_combined_preview(gray, edges, paths_mm, info_lines, last_output_path, rotate_index):
    panel_w = 430
    panel_h = 430

    original_panel = make_original_panel(gray, panel_w, panel_h, rotate_index)
    edge_panel = make_edge_panel(edges, panel_w, panel_h)
    gcode_panel = make_gcode_page_panel(paths_mm, panel_w, panel_h)

    top = np.hstack([original_panel, edge_panel, gcode_panel])

    info_h = 190
    info = np.full((info_h, top.shape[1], 3), 245, dtype=np.uint8)

    y = 24
    for line in info_lines:
        cv2.putText(
            info,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 0, 0),
            2
        )
        y += 23

    cv2.putText(
        info,
        "Keys: a=auto gradient, d=auto density, g=generate G-code, q/ESC=quit",
        (20, 174),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (0, 0, 160),
        2
    )

    if last_output_path:
        text = "Saved: " + last_output_path
        if len(text) > 105:
            text = text[:102] + "..."

        cv2.putText(
            info,
            text,
            (650, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 100, 0),
            1
        )

    combined = np.vstack([top, info])

    return combined


# =========================
# Main
# =========================

def main():
    image_path = select_image_file()

    if not image_path:
        print("No image selected.")
        return

    original_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if original_gray is None:
        print("Cannot read image.")
        return

    initial_gray = rotate_image(original_gray, DEFAULT_ROTATE_INDEX)

    initial_blur_size = DEFAULT_BLUR_RAW * 2 + 1
    auto_low, auto_high = auto_canny_thresholds_by_gradient(initial_gray, initial_blur_size)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("Canny low", WINDOW_NAME, auto_low, 500, nothing)
    cv2.createTrackbar("Canny high", WINDOW_NAME, auto_high, 500, nothing)
    cv2.createTrackbar("Blur", WINDOW_NAME, DEFAULT_BLUR_RAW, 10, nothing)
    cv2.createTrackbar("Min length", WINDOW_NAME, DEFAULT_MIN_LENGTH, 500, nothing)
    cv2.createTrackbar("Simplify x10", WINDOW_NAME, DEFAULT_SIMPLIFY_X10, 100, nothing)
    cv2.createTrackbar("Close", WINDOW_NAME, DEFAULT_CLOSE, 5, nothing)
    cv2.createTrackbar("Size %", WINDOW_NAME, DEFAULT_SIZE_PERCENT, 100, nothing)
    cv2.createTrackbar("Rotate 90", WINDOW_NAME, DEFAULT_ROTATE_INDEX, 3, nothing)
    cv2.createTrackbar("Travel F", WINDOW_NAME, DEFAULT_TRAVEL_FEED, MAX_FEED_RATE, nothing)
    cv2.createTrackbar("Draw F", WINDOW_NAME, DEFAULT_DRAW_FEED, MAX_FEED_RATE, nothing)

    print("Image:", os.path.abspath(image_path))
    print()
    print("Paper settings:")
    print("A4 landscape =", PAPER_WIDTH_MM, "x", PAPER_HEIGHT_MM, "mm")
    print("Origin        = lower-left corner of A4")
    print("Safe range    = X", SAFE_MIN_X, "~", SAFE_MAX_X, ", Y", SAFE_MIN_Y, "~", SAFE_MAX_Y)
    print("Note          = Before running, place the pen at the lower-left corner of the A4 sheet.")
    print("                G92 X0 Y0 will treat that position as the origin.")
    print()
    print("Speed settings:")
    print("Travel F = pen-up moving speed, usually mm/min")
    print("Draw F   = pen-down drawing speed, usually mm/min")
    print("Minimum feed rate is clamped to", MIN_FEED_RATE)
    print()
    print("Rotation trackbar:")
    print("Rotate 90 = 0: 0 deg")
    print("Rotate 90 = 1: 90 deg clockwise")
    print("Rotate 90 = 2: 180 deg")
    print("Rotate 90 = 3: 90 deg counterclockwise")
    print()
    print("Controls:")
    print("a: auto Canny threshold by gradient Otsu")
    print("d: auto Canny threshold by edge density")
    print("g: generate G-code")
    print("q or ESC: quit")
    print()
    print("G-code settings:")
    print("PEN_DOWN_CMD:", PEN_DOWN_CMD)
    print("PEN_UP_CMD:  ", PEN_UP_CMD)
    print("DEFAULT_TRAVEL_FEED:", DEFAULT_TRAVEL_FEED)
    print("DEFAULT_DRAW_FEED:  ", DEFAULT_DRAW_FEED)
    print("DWELL:              ", DWELL_AFTER_PEN)
    print()
    print("Initial auto thresholds:")
    print("Canny low:", auto_low)
    print("Canny high:", auto_high)
    print()

    last_output_path = ""

    while True:
        (
            low,
            high,
            blur_size,
            min_len,
            epsilon,
            close_iter,
            size_percent,
            rotate_index,
            travel_feed,
            draw_feed
        ) = get_trackbar_values()

        working_gray = rotate_image(original_gray, rotate_index)

        edges = make_edges(working_gray, low, high, blur_size, close_iter)
        paths_px = extract_paths(edges, min_len, epsilon)

        h, w = working_gray.shape

        raw_paths_mm, scale_mm_per_px, drawing_w_mm, drawing_h_mm = convert_paths_to_mm(
            paths_px,
            w,
            h,
            size_percent
        )

        safe_paths_mm, safe_stats = sanitize_paths_to_paper(raw_paths_mm)

        edge_density = np.count_nonzero(edges) / edges.size

        info_lines = [
            f"Rotate: {rotation_label(rotate_index)} | Canny low: {low} | Canny high: {high} | Blur: {blur_size}",
            f"Min length: {min_len} | Simplify: {epsilon:.1f} | Close: {close_iter} | Edge density: {edge_density * 100:.2f}%",
            f"Raw paths: {len(paths_px)} | Safe paths: {len(safe_paths_mm)} | Size: {size_percent}% | Output: {drawing_w_mm:.1f} x {drawing_h_mm:.1f} mm",
            f"Speed: Travel F{travel_feed} | Draw F{draw_feed} | Draw speed approx {draw_feed / 60.0:.1f} mm/s",
            f"Safety: clipped segs {safe_stats['clipped_segments']} | dropped segs {safe_stats['dropped_segments']} | out pts {safe_stats['out_of_bounds_points']}"
        ]

        preview = make_combined_preview(
            working_gray,
            edges,
            safe_paths_mm,
            info_lines,
            last_output_path,
            rotate_index
        )

        cv2.imshow(WINDOW_NAME, preview)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("a"):
            blur_raw = cv2.getTrackbarPos("Blur", WINDOW_NAME)
            blur_size_for_auto = blur_raw * 2 + 1
            rotate_index_for_auto = cv2.getTrackbarPos("Rotate 90", WINDOW_NAME)
            auto_gray = rotate_image(original_gray, rotate_index_for_auto)

            auto_low, auto_high = auto_canny_thresholds_by_gradient(auto_gray, blur_size_for_auto)

            cv2.setTrackbarPos("Canny low", WINDOW_NAME, auto_low)
            cv2.setTrackbarPos("Canny high", WINDOW_NAME, auto_high)

            print("Auto threshold by gradient Otsu")
            print("Rotation:", rotation_label(rotate_index_for_auto))
            print("Canny low:", auto_low)
            print("Canny high:", auto_high)
            print()

        elif key == ord("d"):
            blur_raw = cv2.getTrackbarPos("Blur", WINDOW_NAME)
            blur_size_for_auto = blur_raw * 2 + 1
            rotate_index_for_auto = cv2.getTrackbarPos("Rotate 90", WINDOW_NAME)
            auto_gray = rotate_image(original_gray, rotate_index_for_auto)

            auto_low, auto_high = auto_canny_thresholds_by_edge_density(
                auto_gray,
                blur_size_for_auto,
                target_density=DENSITY_TARGET
            )

            cv2.setTrackbarPos("Canny low", WINDOW_NAME, auto_low)
            cv2.setTrackbarPos("Canny high", WINDOW_NAME, auto_high)

            print("Auto threshold by edge density")
            print("Rotation:", rotation_label(rotate_index_for_auto))
            print("Target density:", DENSITY_TARGET)
            print("Canny low:", auto_low)
            print("Canny high:", auto_high)
            print()

        elif key == ord("g"):
            if not validate_paths_in_paper(safe_paths_mm):
                print("Blocked: unsafe path still exists after clipping.")
                continue

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            image_dir = os.path.dirname(os.path.abspath(image_path))
            output_path = os.path.join(image_dir, base_name + ".gcode")

            try:
                abs_output_path = save_gcode(
                    safe_paths_mm,
                    output_path,
                    travel_feed,
                    draw_feed
                )
                last_output_path = abs_output_path

                print("G-code generated.")
                print("Output file:")
                print(abs_output_path)
                print("Rotation:", rotation_label(rotate_index))
                print("Travel feed:", travel_feed)
                print("Draw feed:", draw_feed)
                print(f"Draw speed approx: {draw_feed / 60.0:.2f} mm/s")
                print("Raw path count:", len(raw_paths_mm))
                print("Safe path count:", len(safe_paths_mm))
                print("Input segments:", safe_stats["input_segments"])
                print("Output segments:", safe_stats["output_segments"])
                print("Clipped segments:", safe_stats["clipped_segments"])
                print("Dropped segments:", safe_stats["dropped_segments"])
                print("Out-of-bounds points:", safe_stats["out_of_bounds_points"])
                print(f"Drawing size before clipping: {drawing_w_mm:.2f} x {drawing_h_mm:.2f} mm")
                print(f"Size percent: {size_percent}%")
                print()

            except ValueError as e:
                print("G-code generation blocked by safety check.")
                print(e)
                print()

        elif key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
