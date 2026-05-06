from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class CornerPosition:
    x: int
    y: int


class HarrisAndSiftDetector:
    def __init__(self, image_path: Path) -> None:
        self._image_path: Path = image_path

        self._corners_count: int = 4
        self._quality_level: float = 0.01
        self._min_distance: int = 100
        self._block_size: int = 3
        self._harris_k: float = 0.04

        self._sift_features_count: int = 200

    def _load_image(self) -> np.ndarray:
        image: np.ndarray | None = cv.imread(str(self._image_path))

        if image is None:
            raise FileNotFoundError(f"Could not load image: {self._image_path}")

        return image

    def _convert_to_gray(self, image: np.ndarray) -> np.ndarray:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def _find_harris_corners(self, gray_image: np.ndarray) -> list[CornerPosition]:
        corners: np.ndarray | None = cv.goodFeaturesToTrack(
            image=gray_image,
            maxCorners=self._corners_count,
            qualityLevel=self._quality_level,
            minDistance=self._min_distance,
            blockSize=self._block_size,
            useHarrisDetector=True,
            k=self._harris_k,
        )

        if corners is None:
            return []

        result: list[CornerPosition] = []

        for corner in corners:
            x_raw: float
            y_raw: float
            x_raw, y_raw = corner.ravel()

            result.append(
                CornerPosition(
                    x=round(x_raw),
                    y=round(y_raw),
                )
            )

        return result

    def _find_sift_keypoints(self, gray_image: np.ndarray) -> tuple[cv.KeyPoint, ...]:
        sift = cv.SIFT.create(nfeatures=self._sift_features_count)

        keypoints, _descriptors = sift.detectAndCompute(gray_image, None)

        return tuple(keypoints)

    def _draw_harris_corners(
        self,
        image: np.ndarray,
        corners: list[CornerPosition],
    ) -> np.ndarray:
        result: np.ndarray = image.copy()

        for corner in corners:
            cv.circle(
                img=result,
                center=(corner.x, corner.y),
                radius=8,
                color=(255, 255, 255),
                thickness=3,
                lineType=cv.LINE_AA,
            )

            cv.circle(
                img=result,
                center=(corner.x, corner.y),
                radius=4,
                color=(0, 0, 0),
                thickness=-1,
                lineType=cv.LINE_AA,
            )

            cv.circle(
                img=result,
                center=(corner.x, corner.y),
                radius=1,
                color=(255, 255, 255),
                thickness=-1,
                lineType=cv.LINE_AA,
            )

        return result

    def _draw_sift_keypoints(
        self,
        image: np.ndarray,
        keypoints: tuple[cv.KeyPoint, ...],
    ) -> np.ndarray:
        result: np.ndarray = image.copy()

        for keypoint in keypoints:
            x: int = round(keypoint.pt[0])
            y: int = round(keypoint.pt[1])

            cv.circle(
                img=result,
                center=(x, y),
                radius=3,
                color=(0, 255, 0),
                thickness=-1,
                lineType=cv.LINE_AA,
            )

        return result

    def _show_result(
        self,
        original_image: np.ndarray,
        harris_result: np.ndarray,
        sift_result: np.ndarray,
    ) -> None:
        cv.imshow("Original image", original_image)
        cv.imshow("Harris: 4 strongest corners", harris_result)
        cv.imshow("SIFT keypoints", sift_result)

        cv.waitKey(0)
        cv.destroyAllWindows()

    def run(self) -> None:
        image: np.ndarray = self._load_image()
        gray_image: np.ndarray = self._convert_to_gray(image)

        harris_corners: list[CornerPosition] = self._find_harris_corners(gray_image)
        harris_result: np.ndarray = self._draw_harris_corners(
            image=image,
            corners=harris_corners,
        )

        sift_keypoints: tuple[cv.KeyPoint, ...] = self._find_sift_keypoints(gray_image)
        sift_result: np.ndarray = self._draw_sift_keypoints(
            image=image,
            keypoints=sift_keypoints,
        )

        self._show_result(
            original_image=image,
            harris_result=harris_result,
            sift_result=sift_result,
        )


if __name__ == "__main__":
    detector: HarrisAndSiftDetector = HarrisAndSiftDetector(
        image_path=Path("/Users/admin/Pycharm/MachineVision/project3/data/photo_1.jpg")
    )

    detector.run()
