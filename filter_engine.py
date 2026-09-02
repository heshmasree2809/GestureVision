import cv2
import numpy as np


# =============================================================
# FILTER NAMES
# =============================================================

FILTER_NONE = "NONE"

FILTER_THERMAL = "THERMAL"

FILTER_NEON = "NEON"

FILTER_NIGHT_VISION = "NIGHT VISION"

FILTER_XRAY = "XRAY"

FILTER_CYBERPUNK = "CYBERPUNK"

FILTER_SPIDERVERSE = "SPIDER-VERSE"


# =============================================================
# FILTER LIST
# =============================================================

FILTERS = [
    FILTER_THERMAL,
    FILTER_NEON,
    FILTER_NIGHT_VISION,
    FILTER_XRAY,
    FILTER_CYBERPUNK,
    FILTER_SPIDERVERSE,
]


# =============================================================
# NEXT FILTER
# =============================================================

def get_next_filter(
    current_filter
):

    if current_filter not in FILTERS:

        return FILTERS[0]

    index = FILTERS.index(
        current_filter
    )

    next_index = (
        index + 1
    ) % len(FILTERS)

    return FILTERS[next_index]


# =============================================================
# THERMAL
# =============================================================

def thermal_filter(
    image
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.applyColorMap(
        gray,
        cv2.COLORMAP_INFERNO
    )


# =============================================================
# NEON
# =============================================================

def neon_filter(
    image
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        80,
        160
    )

    edges = cv2.dilate(
        edges,
        None,
        iterations=1
    )

    edges = cv2.GaussianBlur(
        edges,
        (3, 3),
        0
    )

    neon = np.zeros_like(
        image
    )

    neon[:, :, 1] = edges
    neon[:, :, 2] = edges

    return cv2.addWeighted(
        image,
        0.35,
        neon,
        1.2,
        0
    )


# =============================================================
# NIGHT VISION
# =============================================================

def night_vision_filter(
    image
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(
        gray
    )

    result = np.zeros_like(
        image
    )

    result[:, :, 1] = gray

    return result


# =============================================================
# XRAY
# =============================================================

def xray_filter(
    image
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    inverted = cv2.bitwise_not(
        gray
    )

    result = cv2.cvtColor(
        inverted,
        cv2.COLOR_GRAY2BGR
    )

    return result


# =============================================================
# CYBERPUNK
# =============================================================

def cyberpunk_filter(
    image
):

    b, g, r = cv2.split(
        image
    )

    result = np.zeros_like(
        image
    )

    result[:, :, 0] = cv2.add(
        b,
        g // 3
    )

    result[:, :, 1] = cv2.add(
        g,
        r // 4
    )

    result[:, :, 2] = cv2.add(
        r,
        b // 2
    )

    return result


# =============================================================
# SPIDER-VERSE
# =============================================================

def spiderverse_filter(
    image
):

    # ---------------------------------------------------------
    # Slight posterization
    # ---------------------------------------------------------

    small = cv2.resize(
        image,
        None,
        fx=0.5,
        fy=0.5,
        interpolation=cv2.INTER_LINEAR
    )

    small = cv2.resize(
        small,
        (
            image.shape[1],
            image.shape[0]
        ),
        interpolation=cv2.INTER_NEAREST
    )

    # ---------------------------------------------------------
    # Edge extraction
    # ---------------------------------------------------------

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        70,
        140
    )

    edges = cv2.cvtColor(
        edges,
        cv2.COLOR_GRAY2BGR
    )

    result = cv2.addWeighted(
        small,
        0.85,
        edges,
        0.35,
        0
    )

    return result


# =============================================================
# APPLY FILTER TO RECTANGLE
# =============================================================

def apply_portal_filter(
    frame,
    center,
    width,
    height,
    filter_name
):

    if (
        filter_name == FILTER_NONE
    ):

        return frame

    frame_height, frame_width = (
        frame.shape[:2]
    )

    cx, cy = center

    half_width = int(
        width / 2
    )

    half_height = int(
        height / 2
    )

    x1 = max(
        0,
        cx - half_width
    )

    y1 = max(
        0,
        cy - half_height
    )

    x2 = min(
        frame_width,
        cx + half_width
    )

    y2 = min(
        frame_height,
        cy + half_height
    )

    if (
        x2 <= x1
        or
        y2 <= y1
    ):

        return frame

    # ---------------------------------------------------------
    # Crop
    # ---------------------------------------------------------

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:

        return frame

    # ---------------------------------------------------------
    # Select filter
    # ---------------------------------------------------------

    if filter_name == FILTER_THERMAL:

        filtered = thermal_filter(
            crop
        )

    elif filter_name == FILTER_NEON:

        filtered = neon_filter(
            crop
        )

    elif filter_name == FILTER_NIGHT_VISION:

        filtered = night_vision_filter(
            crop
        )

    elif filter_name == FILTER_XRAY:

        filtered = xray_filter(
            crop
        )

    elif filter_name == FILTER_CYBERPUNK:

        filtered = cyberpunk_filter(
            crop
        )

    elif filter_name == FILTER_SPIDERVERSE:

        filtered = spiderverse_filter(
            crop
        )

    else:

        filtered = crop

    # ---------------------------------------------------------
    # Put filtered image back
    # ---------------------------------------------------------

    frame[
        y1:y2,
        x1:x2
    ] = filtered

    return frame