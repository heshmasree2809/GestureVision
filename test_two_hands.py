import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
MODEL_PATH = "models/hand_landmarker.task"
def main():

    print("Starting two-hand diagnostic...")
    print("Show BOTH hands to the camera.")
    print("Press Q to quit.")
    print()

    # ------------------------------------------------------
    # MediaPipe
    # ------------------------------------------------------

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,

        # VERY IMPORTANT
        num_hands=2,

        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    detector = (
        vision.HandLandmarker.create_from_options(
            options
        )
    )

    # ---------------------------------------------------------
    # Camera
    # ---------------------------------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("Could not open camera.")
        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    timestamp = 0

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(
            frame,
            1
        )

        # -----------------------------------------------------
        # BGR -> RGB
        # -----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp += 33

        # -----------------------------------------------------
        # Detect
        # -----------------------------------------------------

        result = detector.detect_for_video(
            image,
            timestamp
        )

        hand_count = len(
            result.hand_landmarks
        )

        # -----------------------------------------------------
        # Print result
        # -----------------------------------------------------

        print(
            f"\rHands detected: {hand_count}",
            end=""
        )

        # -----------------------------------------------------
        # Draw every detected hand
        # -----------------------------------------------------

        for hand_index, landmarks in enumerate(
            result.hand_landmarks
        ):

            # Palm
            wrist = landmarks[0]

            x = int(
                wrist.x *
                frame.shape[1]
            )

            y = int(
                wrist.y *
                frame.shape[0]
            )

            # Handedness
            side = "UNKNOWN"

            if (
                hand_index <
                len(result.handedness)
            ):

                side = (
                    result
                    .handedness[
                        hand_index
                    ][0]
                    .category_name
                )

            # -------------------------------------------------
            # Draw wrist
            # -------------------------------------------------

            cv2.circle(
                frame,
                (x, y),
                12,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                f"{hand_index}: {side}",
                (
                    x + 15,
                    y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # -------------------------------------------------
            # Draw index fingertip
            # -------------------------------------------------

            index_tip = landmarks[8]

            ix = int(
                index_tip.x *
                frame.shape[1]
            )

            iy = int(
                index_tip.y *
                frame.shape[0]
            )

            cv2.circle(
                frame,
                (ix, iy),
                8,
                (255, 0, 255),
                -1
            )

            # -------------------------------------------------
            # Draw all landmarks
            # -------------------------------------------------

            for landmark in landmarks:

                lx = int(
                    landmark.x *
                    frame.shape[1]
                )

                ly = int(
                    landmark.y *
                    frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (lx, ly),
                    3,
                    (255, 255, 255),
                    -1
                )

        # -----------------------------------------------------
        # UI
        # -----------------------------------------------------

        cv2.putText(
            frame,
            f"HANDS: {hand_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "Two Hand Diagnostic",
            frame
        )

        # -----------------------------------------------------
        # Quit
        # -----------------------------------------------------

        if (
            cv2.waitKey(1)
            &
            0xFF
        ) == ord("q"):

            break

    camera.release()

    cv2.destroyAllWindows()

    detector.close()

    print()
    print()
    print("Diagnostic finished.")


if __name__ == "__main__":

    main()
