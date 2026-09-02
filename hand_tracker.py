import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:

    def __init__(
        self,
        model_path="models/hand_landmarker.task",
        max_hands=2,
        detection_confidence=0.5,
        presence_confidence=0.5,
        tracking_confidence=0.5,
    ):

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self.detector = (
            vision.HandLandmarker.create_from_options(
                options
            )
        )

        self.timestamp_ms = 0

    # =========================================================
    # DETECT HANDS
    # =========================================================

    def detect(self, frame):

        height, width = frame.shape[:2]

        # -----------------------------------------------------
        # Convert BGR -> RGB
        # -----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # -----------------------------------------------------
        # MediaPipe video timestamp
        # -----------------------------------------------------

        self.timestamp_ms += 33

        result = self.detector.detect_for_video(
            image,
            self.timestamp_ms
        )

        hands = []

        if not result.hand_landmarks:

            return hands

        # =====================================================
        # PROCESS ALL DETECTED HANDS
        # =====================================================

        for hand_index, landmarks in enumerate(
            result.hand_landmarks
        ):

            # =================================================
            # PIXEL LANDMARKS
            # =================================================

            pixel_landmarks = []

            for landmark in landmarks:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                x = max(
                    0,
                    min(
                        width - 1,
                        x
                    )
                )

                y = max(
                    0,
                    min(
                        height - 1,
                        y
                    )
                )

                pixel_landmarks.append(
                    (x, y)
                )

            # =================================================
            # IMPORTANT LANDMARKS
            # =================================================

            wrist = pixel_landmarks[0]

            thumb_tip = pixel_landmarks[4]

            index_tip = pixel_landmarks[8]

            middle_tip = pixel_landmarks[12]

            ring_tip = pixel_landmarks[16]

            pinky_tip = pixel_landmarks[20]

            # =================================================
            # PALM CENTER
            # =================================================

            palm_points = [
                pixel_landmarks[0],
                pixel_landmarks[5],
                pixel_landmarks[9],
                pixel_landmarks[13],
                pixel_landmarks[17],
            ]

            center_x = int(
                sum(
                    point[0]
                    for point in palm_points
                )
                /
                len(palm_points)
            )

            center_y = int(
                sum(
                    point[1]
                    for point in palm_points
                )
                /
                len(palm_points)
            )

            center = (
                center_x,
                center_y
            )

            # =================================================
            # HANDEDNESS
            #
            # IMPORTANT:
            #
            # Do NOT swap MediaPipe's result here.
            # We keep the original classification.
            #
            # The main application will use the hand's
            # physical position when necessary.
            # =================================================

            if (
                hand_index <
                len(result.handedness)
            ):

                handedness = (
                    result
                    .handedness[
                        hand_index
                    ][0]
                    .category_name
                    .upper()
                )

            else:

                handedness = "UNKNOWN"

            # =================================================
            # STORE HAND
            # =================================================

            hands.append(
                {
                    "landmarks": pixel_landmarks,

                    "handedness": handedness,

                    "center": center,

                    "wrist": wrist,

                    "thumb_tip": thumb_tip,

                    "index_tip": index_tip,

                    "middle_tip": middle_tip,

                    "ring_tip": ring_tip,

                    "pinky_tip": pinky_tip,
                }
            )

        # =====================================================
        # IMPORTANT
        #
        # We DON'T trust handedness labels for ordering.
        #
        # Sort based on the actual position on the mirrored
        # camera image.
        #
        # LEFT side of screen  = LEFT
        # RIGHT side of screen = RIGHT
        # =====================================================

        if len(hands) == 2:

            hands.sort(
                key=lambda hand:
                hand["center"][0]
            )

            hands[0]["screen_side"] = "LEFT"

            hands[1]["screen_side"] = "RIGHT"

        else:

            for hand in hands:

                x = hand["center"][0]

                if x < width / 2:

                    hand["screen_side"] = "LEFT"

                else:

                    hand["screen_side"] = "RIGHT"

        return hands

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        self.detector.close()