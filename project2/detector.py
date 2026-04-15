from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import cv2 as cv
import numpy as np


class SizeLabel(StrEnum):
    SMALL = "small"
    BIG = "big"


class PositionLabel(StrEnum):
    INSIDE = "inside"
    OUTSIDE = "outside"


@dataclass(frozen=True, slots=True, kw_only=True)
class CoinDetection:
    centre_x: int
    centre_y: int
    radius: int
    size_label: SizeLabel
    position_label: PositionLabel


@dataclass(frozen=True, slots=True, kw_only=True)
class TrayDetection:
    contour: np.ndarray
    mask: np.ndarray
    area: float


@dataclass(frozen=True, slots=True, kw_only=True)
class TrayAnalysisResult:
    tray: TrayDetection
    coins: list[CoinDetection]
    small_inside_count: int
    small_outside_count: int
    big_inside_count: int
    big_outside_count: int


class CoinTrayAnalyser:
    def __init__(self) -> None:
        self._tray_low_hsv: np.ndarray = np.array([8, 150, 170], dtype=np.uint8)
        self._tray_high_hsv: np.ndarray = np.array([25, 255, 255], dtype=np.uint8)

        self._tray_kernel_open: np.ndarray = cv.getStructuringElement(
            cv.MORPH_RECT, (5, 5)
        )
        self._tray_kernel_close: np.ndarray = cv.getStructuringElement(
            cv.MORPH_RECT, (21, 21)
        )
        self._tray_kernel_erode: np.ndarray = cv.getStructuringElement(
            cv.MORPH_RECT, (30, 30)
        )

        self._coin_hough_dp: float = 1.05
        self._coin_hough_min_dist: float = 30.0
        self._coin_hough_param1: float = 90.0
        self._coin_hough_param2: float = 35.0
        self._coin_min_radius: int = 10
        self._coin_max_radius: int = 45

    def analyse_image(self, image_path: Path) -> np.ndarray:
        if not image_path.exists():
            raise FileNotFoundError(f"File not found: {image_path}")

        frame: np.ndarray | None = cv.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not decode image at {image_path}")

        tray_detection: TrayDetection = self._detect_tray(frame)
        coin_candidates: list[tuple[int, int, int]] = self._detect_coins(
            tray_detection, frame
        )

        classified_coins: list[CoinDetection] = self._classify_coins(
            coin_candidates=coin_candidates,
            tray_mask=tray_detection.mask,
        )

        result: TrayAnalysisResult = self._build_result(
            tray_detection=tray_detection,
            coins=classified_coins,
        )

        annotated: np.ndarray = frame.copy()
        self._draw_tray(annotated, tray_detection)
        self._draw_coins(annotated, classified_coins)
        self._draw_stats(annotated, result)

        return annotated

    def show_image(self, image_path: Path) -> None:
        annotated: np.ndarray = self.analyse_image(image_path)

        cv.namedWindow(image_path.name, cv.WINDOW_NORMAL)
        cv.imshow(image_path.name, annotated)
        cv.waitKey(0)
        cv.destroyWindow(image_path.name)

    def _detect_tray(self, frame: np.ndarray) -> TrayDetection:
        hsv_frame: np.ndarray = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        tray_mask: np.ndarray = self._get_tray_mask(hsv_frame)
        tray_contour: np.ndarray = self._get_tray_contour(tray_mask)

        tray_mask_filled: np.ndarray = np.zeros(tray_mask.shape, dtype=np.uint8)
        cv.drawContours(tray_mask_filled, [tray_contour], -1, 255, -1)

        tray_area: float = cv.contourArea(tray_contour)

        return TrayDetection(
            contour=tray_contour,
            mask=tray_mask_filled,
            area=tray_area,
        )

    def _get_tray_mask(self, hsv_frame: np.ndarray) -> np.ndarray:
        raw_mask: np.ndarray = cv.inRange(
            hsv_frame, self._tray_low_hsv, self._tray_high_hsv
        )

        mask_open: np.ndarray = cv.morphologyEx(
            raw_mask,
            cv.MORPH_OPEN,
            self._tray_kernel_open,
        )
        mask_close: np.ndarray = cv.morphologyEx(
            mask_open,
            cv.MORPH_CLOSE,
            self._tray_kernel_close,
        )
        mask_erode: np.ndarray = cv.morphologyEx(
            mask_close,
            cv.MORPH_ERODE,
            self._tray_kernel_close,
        )

        return mask_erode

    def _get_tray_contour(self, tray_mask: np.ndarray) -> np.ndarray:
        contours: tuple[np.ndarray, ...]
        contours, _ = cv.findContours(
            tray_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise RuntimeError("Could not find tray contour")

        return max(contours, key=cv.contourArea)

    def _detect_coins(
        self, tray: TrayDetection, frame: np.ndarray
    ) -> list[tuple[int, int, int]]:
        gray_frame: np.ndarray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred: np.ndarray = cv.GaussianBlur(gray_frame, (9, 9), 2.0)

        circles: np.ndarray | None = cv.HoughCircles(
            blurred,
            cv.HOUGH_GRADIENT,
            dp=self._coin_hough_dp,
            minDist=self._coin_hough_min_dist,
            param1=self._coin_hough_param1,
            param2=self._coin_hough_param2,
            minRadius=self._coin_min_radius,
            maxRadius=self._coin_max_radius,
        )

        if circles is None:
            return []

        rounded: np.ndarray = np.round(circles[0]).astype(np.int32)

        detections: list[tuple[int, int, int]] = []
        for circle in rounded:
            centre_x: int = int(circle[0])
            centre_y: int = int(circle[1])
            radius: int = int(circle[2])

            circle_mask: np.ndarray = np.zeros(tray.mask.shape, dtype=np.uint8)
            cv.circle(circle_mask, (centre_x, centre_y), radius, 255, -1)

            overlap_mask: np.ndarray = cv.bitwise_and(circle_mask, tray.mask)

            circle_area: int = cv.countNonZero(circle_mask)
            overlap_area: int = cv.countNonZero(overlap_mask)

            if 0 < overlap_area < circle_area:
                continue

            detections.append((centre_x, centre_y, radius))

        return detections

    def _classify_coins(
        self,
        coin_candidates: list[tuple[int, int, int]],
        tray_mask: np.ndarray,
    ) -> list[CoinDetection]:
        if not coin_candidates:
            return []

        sorted_coin_candidates: list[tuple[int, int, int]] = sorted(
            coin_candidates,
            key=lambda item: item[2],
        )

        split_index: int = self._get_coin_size_split_index(sorted_coin_candidates)

        classified: list[CoinDetection] = []

        for i, (centre_x, centre_y, radius) in enumerate(sorted_coin_candidates):
            size_label: SizeLabel = (
                SizeLabel.SMALL if i < split_index else SizeLabel.BIG
            )
            position_label: PositionLabel = (
                PositionLabel.INSIDE
                if tray_mask[centre_y, centre_x] > 0
                else PositionLabel.OUTSIDE
            )

            classified.append(
                CoinDetection(
                    centre_x=centre_x,
                    centre_y=centre_y,
                    radius=radius,
                    size_label=size_label,
                    position_label=position_label,
                )
            )

        return classified

    def _get_coin_size_split_index(
        self,
        sorted_coin_candidates: list[tuple[int, int, int]],
    ) -> int:
        if len(sorted_coin_candidates) < 2:
            return len(sorted_coin_candidates)

        max_diff: int = -1
        max_diff_at_index: int = len(sorted_coin_candidates)

        for i in range(1, len(sorted_coin_candidates)):
            prev_radius: int = sorted_coin_candidates[i - 1][2]
            radius: int = sorted_coin_candidates[i][2]

            diff: int = radius - prev_radius
            if diff > max_diff:
                max_diff = diff
                max_diff_at_index = i

        if max_diff <= 0:
            return len(sorted_coin_candidates)

        return max_diff_at_index

    def _build_result(
        self,
        tray_detection: TrayDetection,
        coins: list[CoinDetection],
    ) -> TrayAnalysisResult:
        small_inside_count: int = 0
        small_outside_count: int = 0
        big_inside_count: int = 0
        big_outside_count: int = 0

        for coin in coins:
            if (
                coin.size_label == SizeLabel.SMALL
                and coin.position_label == PositionLabel.INSIDE
            ):
                small_inside_count += 1
            elif (
                coin.size_label == SizeLabel.SMALL
                and coin.position_label == PositionLabel.OUTSIDE
            ):
                small_outside_count += 1
            elif (
                coin.size_label == SizeLabel.BIG
                and coin.position_label == PositionLabel.INSIDE
            ):
                big_inside_count += 1
            elif (
                coin.size_label == SizeLabel.BIG
                and coin.position_label == PositionLabel.OUTSIDE
            ):
                big_outside_count += 1

        return TrayAnalysisResult(
            tray=tray_detection,
            coins=coins,
            small_inside_count=small_inside_count,
            small_outside_count=small_outside_count,
            big_inside_count=big_inside_count,
            big_outside_count=big_outside_count,
        )

    def _draw_tray(self, frame: np.ndarray, tray_detection: TrayDetection) -> None:
        cv.drawContours(frame, [tray_detection.contour], -1, (0, 255, 0), 3, cv.LINE_AA)

    def _draw_coins(self, frame: np.ndarray, coins: list[CoinDetection]) -> None:
        for coin in coins:
            if coin.size_label == SizeLabel.BIG:
                circle_colour: tuple[int, int, int] = (0, 255, 0)
            else:
                circle_colour = (255, 0, 0)

            if coin.position_label == PositionLabel.INSIDE:
                centre_colour: tuple[int, int, int] = (255, 255, 255)
            else:
                centre_colour = (255, 0, 255)

            cv.circle(
                frame,
                (coin.centre_x, coin.centre_y),
                coin.radius,
                circle_colour,
                2,
                cv.LINE_AA,
            )
            cv.circle(
                frame,
                (coin.centre_x, coin.centre_y),
                4,
                centre_colour,
                -1,
                cv.LINE_AA,
            )

    def _draw_stats(self, frame: np.ndarray, result: TrayAnalysisResult) -> None:
        lines: list[str] = [
            f"Area: {result.tray.area:.1f}",
            f"Small & In: {result.small_inside_count}",
            f"Small & Out: {result.small_outside_count}",
            f"Big & In: {result.big_inside_count}",
            f"Big & Out: {result.big_outside_count}",
        ]

        start_x: int = 15
        start_y: int = 30
        line_step: int = 28

        for index, line in enumerate(lines):
            y: int = start_y + index * line_step

            cv.putText(
                frame,
                line,
                (start_x, y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )


if __name__ == "__main__":
    image_path: Path = Path(
        "/Users/admin/Pycharm/MachineVision/project2/data/tray8.jpg"
    )

    analyser: CoinTrayAnalyser = CoinTrayAnalyser()
    analyser.show_image(image_path)
