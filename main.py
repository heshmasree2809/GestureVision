import cv2
import time
import math

from src.hand_tracker import HandTracker

from src.gesture_detector import (
    detect_gesture,
    GESTURE_INDEX,
    GESTURE_PEACE,
    GESTURE_FIST,
    GESTURE_OPEN,
    GESTURE_PINCH,
    GESTURE_NONE,
)
from src.portal import Portal

from src.filter_engine import (
    apply_portal_filter,
    FILTER_NONE,
    FILTER_THERMAL,
    FILTERS,
    get_next_filter,
)


# =============================================================
# CAMERA
# =============================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

WINDOW_NAME = "GestureFX Portal"


# =============================================================
# PORTAL SIZE
# =============================================================

MIN_SIZE = 100
MAX_SIZE = 500

MIN_HAND_DISTANCE = 60
MAX_HAND_DISTANCE = 500


# =============================================================
# PINCH LOCK
# =============================================================

# Number of consecutive frames required before a pinch
# is considered intentional.

PINCH_CONFIRM_FRAMES = 8


# =============================================================
# DISTANCE BETWEEN TWO POINTS
# =============================================================

def point_distance(point_a, point_b):

    dx = (
        point_a[0]
        -
        point_b[0]
    )

    dy = (
        point_a[1]
        -
        point_b[1]
    )

    return math.sqrt(
        dx * dx +
        dy * dy
    )


# =============================================================
# DISTANCE -> SQUARE SIZE
# =============================================================

def distance_to_size(distance):

    distance = max(
        MIN_HAND_DISTANCE,
        min(
            MAX_HAND_DISTANCE,
            distance
        )
    )

    normalized = (
        distance -
        MIN_HAND_DISTANCE
    ) / (
        MAX_HAND_DISTANCE -
        MIN_HAND_DISTANCE
    )

    size = (
        MIN_SIZE
        +
        normalized *
        (
            MAX_SIZE -
            MIN_SIZE
        )
    )

    return int(size)


# =============================================================
# DRAW INDEX TIP
# =============================================================

def draw_index_tip(
    frame,
    point,
    label
):

    # Outer circle
    cv2.circle(
        frame,
        point,
        12,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Inner point
    cv2.circle(
        frame,
        point,
        5,
        (0, 255, 255),
        -1,
        cv2.LINE_AA
    )

    # Label
    cv2.putText(
        frame,
        label,
        (
            point[0] + 15,
            point[1] - 10
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )


# =============================================================
# MAIN
# =============================================================

def main():

    print()
    print("==========================================")
    print("          GESTUREFX PORTAL")
    print("==========================================")
    print()

    print("CONTROLS")
    print()
    print("INDEX")
    print("  Move portal")
    print()
    print("TWO INDEX FINGERS")
    print("  Distance = portal size")
    print()
    print("PINCH")
    print("  Hold briefly = Lock / Unlock portal")
    print()
    print("PEACE")
    print("  Next filter")
    print()
    print("FIST")
    print("  Filter OFF")
    print()
    print("OPEN HAND")
    print("  Filter ON")
    print()

    print("FILTERS")

    for index, filter_name in enumerate(
        FILTERS,
        start=1
    ):

        print(
            f"  {index}. {filter_name}"
        )

    print()
    print("Press Q to quit.")
    print()

    # =========================================================
    # HAND TRACKER
    # =========================================================

    tracker = HandTracker(
        model_path="models/hand_landmarker.task",
        max_hands=2,
        detection_confidence=0.5,
        presence_confidence=0.5,
        tracking_confidence=0.5,
    )

    # =========================================================
    # CAMERA
    # =========================================================

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        tracker.close()

        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    # =========================================================
    # PORTAL
    # =========================================================

    portal = None

    # =========================================================
    # FILTER
    # =========================================================

    current_filter = FILTER_THERMAL

    filter_enabled = True

    # =========================================================
    # LOCK
    # =========================================================

    portal_locked = False

    # =========================================================
    # GESTURE STATE
    # =========================================================

    previous_gestures = set()

    # =========================================================
    # PINCH STATE
    # =========================================================

    pinch_frames = 0

    pinch_lock_triggered = False

    # =========================================================
    # FPS
    # =========================================================

    previous_time = time.perf_counter()

    fps = 0.0

    # =========================================================
    # MAIN LOOP
    # =========================================================

    while True:

        # =====================================================
        # CAMERA FRAME
        # =====================================================

        success, frame = camera.read()

        if not success:

            print(
                "ERROR: Could not read camera."
            )

            break

        # -----------------------------------------------------
        # Mirror camera
        # -----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        # =====================================================
        # DETECT HANDS
        # =====================================================

        hands = tracker.detect(
            frame
        )

        # =====================================================
        # HAND A / HAND B
        #
        # We deliberately ignore LEFT/RIGHT.
        # =====================================================

        hand_a = None
        hand_b = None

        if len(hands) >= 1:

            hand_a = hands[0]

        if len(hands) >= 2:

            hand_b = hands[1]

        # =====================================================
        # DETECT GESTURES
        # =====================================================

        current_gestures = set()

        for hand in hands:

            gesture = detect_gesture(
                hand
            )

            current_gestures.add(
                gesture
            )

        # =====================================================
        # PEACE = NEXT FILTER
        #
        # Only triggers when Peace starts.
        # =====================================================

        if (
            GESTURE_PEACE
            in current_gestures
            and
            GESTURE_PEACE
            not in previous_gestures
        ):

            current_filter = get_next_filter(
                current_filter
            )

            filter_enabled = True

        # =====================================================
        # FIST = FILTER OFF
        #
        # Only triggers when Fist starts.
        # =====================================================

        if (
            GESTURE_FIST
            in current_gestures
            and
            GESTURE_FIST
            not in previous_gestures
        ):

            filter_enabled = False

        # =====================================================
        # OPEN HAND = FILTER ON
        #
        # Only triggers when Open starts.
        # =====================================================

        if (
            GESTURE_OPEN
            in current_gestures
            and
            GESTURE_OPEN
            not in previous_gestures
        ):

            filter_enabled = True

        # =====================================================
        # PINCH = LOCK / UNLOCK
        #
        # The pinch must remain stable for several frames.
        #
        # One continuous pinch can only toggle once.
        # =====================================================

        if GESTURE_PINCH in current_gestures:

            pinch_frames += 1

        else:

            # Pinch released.

            pinch_frames = 0

            pinch_lock_triggered = False

        # -----------------------------------------------------
        # Confirm deliberate pinch
        # -----------------------------------------------------

        if (
            pinch_frames >= PINCH_CONFIRM_FRAMES
            and
            not pinch_lock_triggered
        ):

            portal_locked = not portal_locked

            pinch_lock_triggered = True

        # =====================================================
        # CREATE PORTAL
        # =====================================================

        if (
            portal is None
            and
            hand_a is not None
        ):

            index_tip = hand_a[
                "index_tip"
            ]

            portal = Portal(
                center=index_tip,
                width=220,
                height=220,
                handedness="CONTROL"
            )

        # =====================================================
        # MOVE PORTAL
        #
        # Only when unlocked.
        # =====================================================

        if (
            portal is not None
            and
            hand_a is not None
            and
            not portal_locked
        ):

            index_tip = hand_a[
                "index_tip"
            ]

            portal.update(
                center=index_tip,
                width=portal.width,
                height=portal.height
            )

        # =====================================================
        # TWO-HAND SIZE
        #
        # Distance between the two index fingertips controls
        # the square size.
        #
        # Only works while unlocked.
        # =====================================================

        two_hand_size = False

        hand_distance = 0

        if (
            hand_a is not None
            and
            hand_b is not None
            and
            not portal_locked
        ):

            index_a = hand_a[
                "index_tip"
            ]

            index_b = hand_b[
                "index_tip"
            ]

            # -------------------------------------------------
            # Distance
            # -------------------------------------------------

            hand_distance = point_distance(
                index_a,
                index_b
            )

            # -------------------------------------------------
            # Convert distance to square size
            # -------------------------------------------------

            new_size = distance_to_size(
                hand_distance
            )

            two_hand_size = True

            if portal is not None:

                portal.update(
                    center=portal.center,
                    width=new_size,
                    height=new_size
                )

        # =====================================================
        # APPLY FILTER
        # =====================================================

        if (
            filter_enabled
            and
            portal is not None
            and
            current_filter != FILTER_NONE
        ):

            frame = apply_portal_filter(
                frame,
                portal.center,
                portal.width,
                portal.height,
                current_filter
            )

        # =====================================================
        # DRAW PORTAL
        # =====================================================

        if portal is not None:

            portal.draw(
                frame
            )

        # =====================================================
        # DRAW TWO INDEX TIPS
        #
        # Draw AFTER the portal/filter so they stay visible.
        # =====================================================

        if (
            hand_a is not None
            and
            hand_b is not None
            and
            not portal_locked
        ):

            index_a = hand_a[
                "index_tip"
            ]

            index_b = hand_b[
                "index_tip"
            ]

            # -------------------------------------------------
            # Tips
            # -------------------------------------------------

            draw_index_tip(
                frame,
                index_a,
                "A"
            )

            draw_index_tip(
                frame,
                index_b,
                "B"
            )

            # -------------------------------------------------
            # Line between tips
            # -------------------------------------------------

            cv2.line(
                frame,
                index_a,
                index_b,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            # -------------------------------------------------
            # Distance text
            # -------------------------------------------------

            middle_x = int(
                (
                    index_a[0]
                    +
                    index_b[0]
                ) / 2
            )

            middle_y = int(
                (
                    index_a[1]
                    +
                    index_b[1]
                ) / 2
            )

            cv2.putText(
                frame,
                str(
                    int(hand_distance)
                ),
                (
                    middle_x,
                    middle_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        # =====================================================
        # FPS
        # =====================================================

        current_time = time.perf_counter()

        delta = (
            current_time -
            previous_time
        )

        previous_time = current_time

        if delta > 0:

            instant_fps = (
                1.0 /
                delta
            )

            fps = (
                fps * 0.9
                +
                instant_fps * 0.1
            )

        # =====================================================
        # STATUS
        # =====================================================

        if filter_enabled:

            filter_text = current_filter

        else:

            filter_text = "OFF"

        if portal_locked:

            lock_text = "LOCKED"

        else:

            lock_text = "UNLOCKED"

        if two_hand_size:

            size_text = (
                f"DISTANCE: "
                f"{int(hand_distance)}"
            )

        else:

            size_text = "DISTANCE: --"

        # =====================================================
        # HEADER
        # =====================================================

        cv2.putText(
            frame,
            "GESTUREFX PORTAL",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"FILTER: {filter_text}",
            (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"PORTAL: {lock_text}",
            (15, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"HANDS: {len(hands)}",
            (15, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            size_text,
            (15, 136),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (15, 162),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # =====================================================
        # CONTROLS
        # =====================================================

        cv2.putText(
            frame,
            "INDEX = MOVE",
            (15, 192),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "2 INDEXES = SIZE",
            (15, 217),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "PINCH = LOCK",
            (15, 242),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "PEACE = NEXT FILTER",
            (15, 267),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "FIST = OFF",
            (15, 292),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "OPEN = ON",
            (15, 317),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # =====================================================
        # DISPLAY
        # =====================================================

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        # =====================================================
        # QUIT
        # =====================================================

        if (
            cv2.waitKey(1)
            &
            0xFF
        ) == ord("q"):

            break

        # =====================================================
        # SAVE CURRENT GESTURES
        # =====================================================

        previous_gestures = current_gestures

    # =========================================================
    # CLEANUP
    # =========================================================

    camera.release()

    cv2.destroyAllWindows()

    tracker.close()

    print()
    print("GestureFX Portal stopped.")
    print()


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    main()
