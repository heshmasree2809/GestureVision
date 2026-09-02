import math


# =============================================================
# GESTURE NAMES
# =============================================================

GESTURE_NONE = "NONE"

GESTURE_INDEX = "INDEX"

GESTURE_PEACE = "PEACE"

GESTURE_FIST = "FIST"

GESTURE_OPEN = "OPEN"

GESTURE_PINCH = "PINCH"


# =============================================================
# DISTANCE
# =============================================================

def distance(point_a, point_b):

    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]

    return math.sqrt(
        dx * dx +
        dy * dy
    )


# =============================================================
# PALM SIZE
#
# We use palm size to make pinch detection scale-independent.
# =============================================================

def get_palm_size(landmarks):

    return distance(
        landmarks[0],   # wrist
        landmarks[9]    # middle MCP
    )


# =============================================================
# FINGER EXTENDED
# =============================================================

def finger_extended(
    landmarks,
    tip,
    pip,
    mcp
):

    tip_distance = distance(
        landmarks[mcp],
        landmarks[tip]
    )

    pip_distance = distance(
        landmarks[mcp],
        landmarks[pip]
    )

    return tip_distance > (
        pip_distance * 1.25
    )


# =============================================================
# FINGER FOLDED
# =============================================================

def finger_folded(
    landmarks,
    tip,
    pip,
    mcp
):

    return not finger_extended(
        landmarks,
        tip,
        pip,
        mcp
    )


# =============================================================
# PINCH
#
# Thumb tip = landmark 4
# Index tip = landmark 8
#
# The threshold is relative to palm size so the pinch works
# at different distances from the camera.
# =============================================================

def is_pinch(landmarks):

    thumb_tip = landmarks[4]

    index_tip = landmarks[8]

    pinch_distance = distance(
        thumb_tip,
        index_tip
    )

    palm_size = get_palm_size(
        landmarks
    )

    if palm_size <= 0:

        return False

    # Fingers are pinched when their tips are very close.
    return pinch_distance < (
        palm_size * 0.45
    )


# =============================================================
# PEACE
# =============================================================

def is_peace(landmarks):

    index_extended = finger_extended(
        landmarks,
        8,
        6,
        5
    )

    middle_extended = finger_extended(
        landmarks,
        12,
        10,
        9
    )

    ring_folded = finger_folded(
        landmarks,
        16,
        14,
        13
    )

    pinky_folded = finger_folded(
        landmarks,
        20,
        18,
        17
    )

    return (
        index_extended
        and
        middle_extended
        and
        ring_folded
        and
        pinky_folded
    )


# =============================================================
# OPEN HAND
# =============================================================

def is_open_hand(landmarks):

    index_extended = finger_extended(
        landmarks,
        8,
        6,
        5
    )

    middle_extended = finger_extended(
        landmarks,
        12,
        10,
        9
    )

    ring_extended = finger_extended(
        landmarks,
        16,
        14,
        13
    )

    pinky_extended = finger_extended(
        landmarks,
        20,
        18,
        17
    )

    return (
        index_extended
        and
        middle_extended
        and
        ring_extended
        and
        pinky_extended
    )


# =============================================================
# FIST
# =============================================================

def is_fist(landmarks):

    index_folded = finger_folded(
        landmarks,
        8,
        6,
        5
    )

    middle_folded = finger_folded(
        landmarks,
        12,
        10,
        9
    )

    ring_folded = finger_folded(
        landmarks,
        16,
        14,
        13
    )

    pinky_folded = finger_folded(
        landmarks,
        20,
        18,
        17
    )

    return (
        index_folded
        and
        middle_folded
        and
        ring_folded
        and
        pinky_folded
    )


# =============================================================
# INDEX
# =============================================================

def is_index(landmarks):

    index_extended = finger_extended(
        landmarks,
        8,
        6,
        5
    )

    middle_folded = finger_folded(
        landmarks,
        12,
        10,
        9
    )

    ring_folded = finger_folded(
        landmarks,
        16,
        14,
        13
    )

    pinky_folded = finger_folded(
        landmarks,
        20,
        18,
        17
    )

    return (
        index_extended
        and
        middle_folded
        and
        ring_folded
        and
        pinky_folded
    )


# =============================================================
# MAIN GESTURE DETECTOR
# =============================================================

def detect_gesture(hand):

    landmarks = hand["landmarks"]

    # ---------------------------------------------------------
    # PINCH FIRST
    #
    # This is important because a pinch can otherwise be
    # interpreted as another partially folded gesture.
    # ---------------------------------------------------------

    if is_pinch(landmarks):

        return GESTURE_PINCH

    # ---------------------------------------------------------
    # PEACE
    # ---------------------------------------------------------

    if is_peace(landmarks):

        return GESTURE_PEACE

    # ---------------------------------------------------------
    # OPEN HAND
    # ---------------------------------------------------------

    if is_open_hand(landmarks):

        return GESTURE_OPEN

    # ---------------------------------------------------------
    # FIST
    # ---------------------------------------------------------

    if is_fist(landmarks):

        return GESTURE_FIST

    # ---------------------------------------------------------
    # INDEX
    # ---------------------------------------------------------

    if is_index(landmarks):

        return GESTURE_INDEX

    # ---------------------------------------------------------
    # Nothing recognized
    # ---------------------------------------------------------

    return GESTURE_NONE