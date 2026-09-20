#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np

EXPECTED_IDS = [80, 85, 90, 95]
LETTERS = "ABCDEFGHIJK"
GRID_COORDINATES = np.arange(1, 12, dtype=np.float32) * 75.0


def order_marker_points(points):
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def detect_markers_and_rectify(image):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)

    if corners is None or ids is None:
        raise RuntimeError('No ArUco markers were detected in the image.')

    marker_map = {}
    for marker_id, marker_corners in zip(ids.flatten(), corners):
        marker_map[int(marker_id)] = order_marker_points(marker_corners[0])

    missing = [mid for mid in EXPECTED_IDS if mid not in marker_map]
    if missing:
        raise RuntimeError(f'Missing required ArUco markers: {missing}')

    src = np.float32([
        marker_map[80][2],
        marker_map[85][3],
        marker_map[90][0],
        marker_map[95][1],
    ])
    dst = np.float32([
        [0, 0],
        [899, 0],
        [899, 899],
        [0, 899],
    ])

    transform = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(image, transform, (900, 900))
    return rectified, sorted(EXPECTED_IDS)


def preprocess(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_centres(mask, min_area=150, min_circularity=0.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centres = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        if min_circularity > 0.0:
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0.0 or 4.0 * np.pi * area / (perimeter * perimeter) < min_circularity:
                continue
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            continue
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        centres.append((cx, cy))
    return centres


def nearest_grid_label(point):
    x, y = point
    col_idx = int(np.argmin(np.abs(GRID_COORDINATES - x)))
    row_idx = int(np.argmin(np.abs(GRID_COORDINATES - y)))
    return f'{LETTERS[col_idx]}{row_idx + 1}'


def detect_survivors(rectified):
    hsv = cv2.cvtColor(rectified, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, np.array([0, 60, 40]), np.array([12, 255, 255]))
    red_mask |= cv2.inRange(hsv, np.array([170, 60, 40]), np.array([180, 255, 255]))
    red_mask = preprocess(red_mask)

    yellow_mask = cv2.inRange(hsv, np.array([18, 60, 40]), np.array([45, 255, 255]))
    yellow_mask = preprocess(yellow_mask)

    critical_points = {nearest_grid_label(point) for point in find_centres(red_mask, min_area=300)}
    stable_points = {
        nearest_grid_label(point)
        for point in find_centres(yellow_mask, min_area=300, min_circularity=0.75)
    }

    return sorted(critical_points), sorted(stable_points)


def write_results(output_path, marker_ids, critical, stable):
    with output_path.open('w', encoding='utf-8') as outfile:
        outfile.write(f'Detected marker IDs: {marker_ids}\n\n')
        outfile.write(f'Critical Survivors: {", ".join(critical) if critical else ""}\n')
        outfile.write(f'Stable Survivors: {", ".join(stable) if stable else ""}\n')


def main():
    parser = argparse.ArgumentParser(description='Task 1A submission script.')
    parser.add_argument('--image', required=True, help='Path to the input arena image.')
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f'Input image not found: {image_path}')

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Could not load image: {image_path}')

    rectified, marker_ids = detect_markers_and_rectify(image)
    critical, stable = detect_survivors(rectified)
    output_path = image_path.with_name(f'{image_path.stem}_results.txt')
    write_results(output_path, marker_ids, critical, stable)

    print(f'Detected marker IDs: {marker_ids}')
    print(f'Critical Survivors: {critical}')
    print(f'Stable Survivors: {stable}')
    print(f'Wrote {output_path}')


if __name__ == '__main__':
    main()
 