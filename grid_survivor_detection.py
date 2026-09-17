#!/usr/bin/env python3

import cv2
import numpy as np
import math


# ============================================================
# 1. CREATE 121 GRID INTERSECTIONS
# ============================================================

def create_intersections():

    columns = "ABCDEFGHIJK"
    intersections = {}

    # 900x900 arena divided into 12 equal cells
    # Therefore each cell = 75 pixels
    cell_size = 900 / 12

    for row in range(11):
        for col in range(11):

            x = (col + 1) * cell_size
            y = (row + 1) * cell_size

            label = f"{columns[col]}{row + 1}"

            intersections[label] = (x, y)

    return intersections


# ============================================================
# 2. FIND NEAREST INTERSECTION
# ============================================================

def assign_nearest_intersection(cx, cy, intersections):

    nearest_label = None
    minimum_distance = float("inf")

    for label, (x, y) in intersections.items():

        distance = math.sqrt(
            (cx - x) ** 2 +
            (cy - y) ** 2
        )

        if distance < minimum_distance:

            minimum_distance = distance
            nearest_label = label

    return nearest_label, minimum_distance


# ============================================================
# 3. FILTER CONTOURS
# ============================================================

def filter_contours(contours, ratio=0.2):

    if not contours:
        return []

    areas = [
        cv2.contourArea(contour)
        for contour in contours
    ]

    median_area = np.median(areas)

    if median_area <= 0:
        return []

    filtered = []

    for contour, area in zip(contours, areas):

        if area >= median_area * ratio:
            filtered.append(contour)

    return filtered


# ============================================================
# 4. FIND CENTRE OF CONTOUR
# ============================================================

def find_contour_centres(contours):

    centres = []

    for contour in contours:

        M = cv2.moments(contour)

        # Prevent division by zero
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        centres.append((cx, cy))

    return centres


# ============================================================
# 5. CREATE RED / YELLOW MASKS
# ============================================================

def create_survivor_masks(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # RED
    # --------------------------------------------------------

    lower_red_1 = np.array([0, 100, 100])
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([170, 100, 100])
    upper_red_2 = np.array([179, 255, 255])

    red_mask_1 = cv2.inRange(
        hsv,
        lower_red_1,
        upper_red_1
    )

    red_mask_2 = cv2.inRange(
        hsv,
        lower_red_2,
        upper_red_2
    )

    red_mask = cv2.bitwise_or(
        red_mask_1,
        red_mask_2
    )

    # --------------------------------------------------------
    # YELLOW
    # --------------------------------------------------------

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    yellow_mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow
    )

    # --------------------------------------------------------
    # MORPHOLOGICAL CLEANING
    # --------------------------------------------------------

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return red_mask, yellow_mask


# ============================================================
# 6. DETECT SURVIVORS
# ============================================================

def detect_survivors(rectified_image):

    # Create masks
    red_mask, yellow_mask = create_survivor_masks(
        rectified_image
    )

    # --------------------------------------------------------
    # Find RED contours
    # --------------------------------------------------------

    red_contours, _ = cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # --------------------------------------------------------
    # Find YELLOW contours
    # --------------------------------------------------------

    yellow_contours, _ = cv2.findContours(
        yellow_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # --------------------------------------------------------
    # Remove small noise
    # --------------------------------------------------------

    red_contours = filter_contours(
        red_contours
    )

    yellow_contours = filter_contours(
        yellow_contours
    )

    # --------------------------------------------------------
    # Find centres
    # --------------------------------------------------------

    red_centres = find_contour_centres(
        red_contours
    )

    yellow_centres = find_contour_centres(
        yellow_contours
    )

    # --------------------------------------------------------
    # Create grid
    # --------------------------------------------------------

    intersections = create_intersections()

    critical_survivors = []
    stable_survivors = []

    # --------------------------------------------------------
    # RED → CRITICAL SURVIVOR
    # --------------------------------------------------------

    for cx, cy in red_centres:

        label, distance = assign_nearest_intersection(
            cx,
            cy,
            intersections
        )

        if label is not None:
            critical_survivors.append(
                label
            )

    # --------------------------------------------------------
    # YELLOW → STABLE SURVIVOR
    # --------------------------------------------------------

    for cx, cy in yellow_centres:

        label, distance = assign_nearest_intersection(
            cx,
            cy,
            intersections
        )

        if label is not None:
            stable_survivors.append(
                label
            )

    return (
        critical_survivors,
        stable_survivors,
        red_centres,
        yellow_centres
    )


# ============================================================
# 7. DEBUG VISUALIZATION
# ============================================================

def draw_debug_image(
    image,
    red_contours,
    yellow_contours,
    red_centres,
    yellow_centres,
    intersections
):

    debug = image.copy()

    # --------------------------------------------------------
    # Draw all 121 grid intersections
    # --------------------------------------------------------

    for label, (x, y) in intersections.items():

        x = int(round(x))
        y = int(round(y))

        cv2.circle(
            debug,
            (x, y),
            3,
            (255, 255, 255),
            -1
        )

    # --------------------------------------------------------
    # Draw RED survivor outlines
    # --------------------------------------------------------

    cv2.drawContours(
        debug,
        red_contours,
        -1,
        (0, 0, 255),
        2
    )

    # --------------------------------------------------------
    # Draw YELLOW survivor outlines
    # --------------------------------------------------------

    cv2.drawContours(
        debug,
        yellow_contours,
        -1,
        (0, 255, 255),
        2
    )

    # --------------------------------------------------------
    # RED centres + labels
    # --------------------------------------------------------

    for cx, cy in red_centres:

        label, distance = assign_nearest_intersection(
            cx,
            cy,
            intersections
        )

        x = int(round(cx))
        y = int(round(cy))

        # Centre dot
        cv2.circle(
            debug,
            (x, y),
            6,
            (0, 0, 255),
            -1
        )

        # Label
        if label is not None:

            cv2.putText(
                debug,
                label,
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    # --------------------------------------------------------
    # YELLOW centres + labels
    # --------------------------------------------------------

    for cx, cy in yellow_centres:

        label, distance = assign_nearest_intersection(
            cx,
            cy,
            intersections
        )

        x = int(round(cx))
        y = int(round(cy))

        # Centre dot
        cv2.circle(
            debug,
            (x, y),
            6,
            (0, 255, 255),
            -1
        )

        # Label
        if label is not None:

            cv2.putText(
                debug,
                label,
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    return debug


# ============================================================
# 8. MAIN TEST
# ============================================================

if __name__ == "__main__":

    image_path = "image_1.jpg"

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise RuntimeError(
            f"Could not load image: {image_path}"
        )

    print("Image loaded successfully")
    print(
        "Image size:",
        image.shape
    )

    # --------------------------------------------------------
    # Detect survivors
    # --------------------------------------------------------

    (
        critical,
        stable,
        red_centres,
        yellow_centres
    ) = detect_survivors(image)

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nCritical Survivors:")
    print(critical)

    print("\nStable Survivors:")
    print(stable)

    # --------------------------------------------------------
    # Print red centres
    # --------------------------------------------------------

    print("\nRed centres:")

    for centre in red_centres:

        print(
            f"({centre[0]:.2f}, "
            f"{centre[1]:.2f})"
        )

    # --------------------------------------------------------
    # Print yellow centres
    # --------------------------------------------------------

    print("\nYellow centres:")

    for centre in yellow_centres:

        print(
            f"({centre[0]:.2f}, "
            f"{centre[1]:.2f})"
        )

    # --------------------------------------------------------
    # Prepare debug contours
    # --------------------------------------------------------

    intersections = create_intersections()

    red_mask, yellow_mask = create_survivor_masks(
        image
    )

    red_contours, _ = cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    yellow_contours, _ = cv2.findContours(
        yellow_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    red_contours = filter_contours(
        red_contours
    )

    yellow_contours = filter_contours(
        yellow_contours
    )

    # --------------------------------------------------------
    # Generate debug image
    # --------------------------------------------------------

    debug_image = draw_debug_image(
        image,
        red_contours,
        yellow_contours,
        red_centres,
        yellow_centres,
        intersections
    )

    # --------------------------------------------------------
    # Save debug image
    # --------------------------------------------------------

    debug_path = "grid_survivor_debug.jpg"

    cv2.imwrite(
        debug_path,
        debug_image
    )

    print("\nDebug image saved as:")
    print(debug_path)
