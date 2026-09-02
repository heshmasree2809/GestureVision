import cv2


class Portal:

    def __init__(
        self,
        center,
        width=220,
        height=220,
        handedness="CONTROL"
    ):

        self.center = center

        self.width = width
        self.height = height

        self.handedness = handedness

        self.locked = False

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        center=None,
        width=None,
        height=None
    ):

        if center is not None:

            self.center = center

        if width is not None:

            self.width = int(width)

        if height is not None:

            self.height = int(height)

    # =========================================================
    # RECTANGLE
    # =========================================================

    def get_rect(self):

        cx, cy = self.center

        half_width = self.width // 2
        half_height = self.height // 2

        x1 = cx - half_width
        y1 = cy - half_height

        x2 = cx + half_width
        y2 = cy + half_height

        return (
            x1,
            y1,
            x2,
            y2
        )

    # =========================================================
    # DRAW
    # =========================================================

    def draw(
        self,
        frame
    ):

        x1, y1, x2, y2 = self.get_rect()

        height, width = frame.shape[:2]

        # Keep rectangle inside screen.

        x1 = max(
            0,
            min(
                width - 1,
                x1
            )
        )

        y1 = max(
            0,
            min(
                height - 1,
                y1
            )
        )

        x2 = max(
            0,
            min(
                width - 1,
                x2
            )
        )

        y2 = max(
            0,
            min(
                height - 1,
                y2
            )
        )

        # -----------------------------------------------------
        # Portal border
        # -----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # -----------------------------------------------------
        # Inner border
        # -----------------------------------------------------

        if (
            x2 - x1 > 8
            and
            y2 - y1 > 8
        ):

            cv2.rectangle(
                frame,
                (
                    x1 + 4,
                    y1 + 4
                ),
                (
                    x2 - 4,
                    y2 - 4
                ),
                (180, 180, 180),
                1,
                cv2.LINE_AA
            )

        # -----------------------------------------------------
        # Lock indicator
        # -----------------------------------------------------

        if self.locked:

            cv2.putText(
                frame,
                "LOCKED",
                (
                    x1 + 10,
                    y1 + 25
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )