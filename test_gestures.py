import cv2

from src.hand_tracker import HandTracker

from src.gesture_detector import (
    detect_gesture,
    GESTURE_NONE,
    GESTURE_INDEX,
    GESTURE_PEACE,
    GESTURE_FIST,
)


def main():

    print()
    print("==========================================")
    print("       GestureFX Gesture Test")
    print("==========================================")
    print()
    print("Try:")
    print("  Peace  = next filter")
    print("  Index  = size control")
    print("  Fist   = filter off")
    print()
    print("Press Q to quit.")
    print()

    tracker = HandTracker(
        model_path="models/hand_landmarker.task",
        max_hands=2,
        detection_confidence=0.5,
        presence_confidence=0.5,
        tracking_confidence=0.5,
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("ERROR: Could not open camera.")

        tracker.close()

        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    last_gesture = GESTURE_NONE

    while True:

        success, frame = camera.read()

        if not success:

            break

        frame = cv2.flip(
            frame,
            1
        )

        hands = tracker.detect(
            frame
        )

        gesture = GESTURE_NONE

        if hands:

            gesture = detect_gesture(
                hands[0]
            )

        # -----------------------------------------------------
        # Only update displayed gesture when recognized
        # -----------------------------------------------------

        if gesture != GESTURE_NONE:

            last_gesture = gesture

        # -----------------------------------------------------
        # Draw landmarks
        # -----------------------------------------------------

        if hands:

            hand = hands[0]

            for point in hand["landmarks"]:

                cv2.circle(
                    frame,
                    point,
                    3,
                    (255, 255, 255),
                    -1,
                    cv2.LINE_AA
                )

        # -----------------------------------------------------
        # Gesture text
        # -----------------------------------------------------

        cv2.putText(
            frame,
            f"Gesture: {last_gesture}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # -----------------------------------------------------
        # Instructions
        # -----------------------------------------------------

        cv2.putText(
            frame,
            "PEACE = FILTER",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "INDEX = SIZE",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "FIST = OFF",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "Q = QUIT",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.imshow(
            "GestureFX - Gesture Test",
            frame
        )

        if (
            cv2.waitKey(1)
            &
            0xFF
        ) == ord("q"):

            break

    camera.release()

    cv2.destroyAllWindows()

    tracker.close()

    print()
    print("Gesture test finished.")
    print()


if __name__ == "__main__":

    main()