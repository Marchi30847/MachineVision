from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from frame_providers import FrameProvider, StaticImageSource, VideoSource


@dataclass(frozen=True, slots=True, kw_only=True)
class BallPosition:
    centre_x: int
    centre_y: int
    radius: int


class RedBallTracker:
    def __init__(self, provide: FrameProvider) -> None:
        self._provider: FrameProvider = provide

        self._current_pos: BallPosition | None = None
        self._last_pos: BallPosition | None = None

        self._kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
        self._kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))

        self._min_contour_area: int = 100

        self._saturation_weight: float = 0.3
        self._value_weight: float = 0.7

    def _get_mask(self, hsv_frame: np.ndarray) -> np.ndarray:
        mask0_low: np.ndarray = np.array([0, 90, 140], dtype=np.uint8)
        mask0_high: np.ndarray = np.array([20, 255, 255], dtype=np.uint8)
        mask0: np.ndarray = cv.inRange(hsv_frame, mask0_low, mask0_high)

        mask1_low: np.ndarray = np.array([160, 90, 140], dtype=np.uint8)
        mask1_high: np.ndarray = np.array([180, 255, 255], dtype=np.uint8)
        mask1: np.ndarray = cv.inRange(hsv_frame, mask1_low, mask1_high)

        combined: np.ndarray = cv.bitwise_or(mask0, mask1)

        mask_clean_open: np.ndarray = cv.morphologyEx(combined, cv.MORPH_OPEN, self._kernel_open)
        mask_clean_close: np.ndarray = cv.morphologyEx(
            mask_clean_open, cv.MORPH_CLOSE, self._kernel_close
        )

        return mask_clean_close

    def _get_contours(self, mask: np.ndarray) -> list[np.ndarray]:
        contours: tuple[np.ndarray, ...]
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        final_contours: list[np.ndarray] = [
            contour for contour in contours if cv.contourArea(contour) > self._min_contour_area
        ]

        return final_contours

    def _draw_ball(self, frame: np.ndarray) -> None:
        if self._current_pos is not None:
            cv.circle(
                frame,
                (self._current_pos.centre_x, self._current_pos.centre_y),
                self._current_pos.radius,
                (0, 255, 0),
                2,
            )
            cv.circle(
                frame, (self._current_pos.centre_x, self._current_pos.centre_y), 2, (0, 0, 255), -1
            )
            cv.putText(
                frame,
                "RED BALL",
                (
                    self._current_pos.centre_x - 25,
                    self._current_pos.centre_y - self._current_pos.radius - 10,
                ),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv.LINE_AA,
            )

        elif self._last_pos is not None:
            cv.circle(
                frame,
                (self._last_pos.centre_x, self._last_pos.centre_y),
                self._last_pos.radius,
                (128, 128, 128),
                1,
                cv.LINE_AA,
            )

            cv.putText(
                frame,
                "LOST RED BALL",
                (
                    self._last_pos.centre_x - 25,
                    self._last_pos.centre_y - self._last_pos.radius - 10,
                ),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (128, 128, 128),
                1,
                cv.LINE_AA,
            )

        else:
            pass

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        hsv_frame: np.ndarray = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask: np.ndarray = self._get_mask(hsv_frame)
        contours: list[np.ndarray] = self._get_contours(mask)

        self._current_pos = None

        if contours:
            best_contour: np.ndarray | None = None
            best_score: float = -1.0

            stencil: np.ndarray = np.zeros(mask.shape, dtype=np.uint8)
            for contour in contours:
                stencil.fill(0)
                cv.drawContours(stencil, [contour], -1, 255, -1)

                mean_hsv: tuple[float, float, float, float] = cv.mean(hsv_frame, mask=stencil)

                score: float = (
                    self._saturation_weight * mean_hsv[1] + self._value_weight * mean_hsv[2]
                )

                if score > best_score:
                    best_score = score
                    best_contour = contour

            (x_f, y_f), radius_f = cv.minEnclosingCircle(best_contour)

            self._current_pos = BallPosition(
                centre_x=int(x_f), centre_y=int(y_f), radius=int(radius_f)
            )
            self._last_pos = self._current_pos

        self._draw_ball(frame)

        return frame

    def run(self) -> None:
        try:
            while self._provider.is_open():
                success: bool
                frame: np.ndarray | None
                success, frame = self._provider.read()

                if not success or frame is None:
                    break

                processed: np.ndarray = self._process_frame(frame)
                cv.imshow("processed", processed)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self._provider.release()
            cv.destroyAllWindows()


if __name__ == "__main__":
    static_image_provider: FrameProvider = StaticImageSource(
        Path("/Users/admin/Pycharm/MachineVision/project1/data/red_ball.jpg")
    )
    videos_provider: FrameProvider = VideoSource(
        Path("/Users/admin/Pycharm/MachineVision/project1/data/rgb_ball_720.mp4")
    )
    # camera_provider: FrameProvider = CameraSource(0)

    tracker: RedBallTracker = RedBallTracker(videos_provider)
    tracker.run()
