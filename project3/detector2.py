from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2 as cv
import numpy as np


class FrameSource(Protocol):
    def is_open(self) -> bool: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


class ImageFileSource:
    def __init__(self, image_path: Path) -> None:
        self._image_path: Path = image_path
        self._was_read: bool = False

    def is_open(self) -> bool:
        return not self._was_read

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._was_read:
            return False, None

        image: np.ndarray | None = cv.imread(str(self._image_path))

        if image is None:
            raise FileNotFoundError(f"Could not load image: {self._image_path}")

        self._was_read = True

        return True, image

    def release(self) -> None:
        pass


class VideoFileSource:
    def __init__(self, video_path: Path) -> None:
        self._video_path: Path = video_path
        self._capture: cv.VideoCapture = cv.VideoCapture(str(video_path))

        if not self._capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

    def is_open(self) -> bool:
        return self._capture.isOpened()

    def read(self) -> tuple[bool, np.ndarray | None]:
        success: bool
        frame: np.ndarray | None

        success, frame = self._capture.read()

        return success, frame

    def release(self) -> None:
        self._capture.release()


@dataclass(frozen=True, slots=True, kw_only=True)
class SiftFeatures:
    keypoints: tuple[cv.KeyPoint, ...]
    descriptors: np.ndarray


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureMatchResult:
    inlier_matches: list[cv.DMatch]
    projected_corners: np.ndarray


class FeatureMatchingTracker:
    def __init__(
        self,
        query_image_path: Path,
        target_source: FrameSource,
    ) -> None:
        self._query_image_path: Path = query_image_path
        self._target_source: FrameSource = target_source

        self._sift_features_count: int = 1800

        self._knn_matches_count: int = 2
        self._lowe_ratio: float = 0.72

        self._min_good_matches: int = 6
        self._min_inlier_matches: int = 8
        self._ransac_reprojection_threshold: float = 3.5

        self._outline_color: tuple[int, int, int] = (0, 255, 0)
        self._last_position_color: tuple[int, int, int] = (128, 128, 128)
        self._outline_thickness: int = 4

        self._min_outline_area_ratio: float = 0.002
        self._max_outline_area_ratio: float = 0.55
        self._max_center_jump_px: float = 220.0

        self._smoothing_alpha: float = 0.35

    def _load_query_image(self) -> np.ndarray:
        image: np.ndarray | None = cv.imread(str(self._query_image_path))

        if image is None:
            raise FileNotFoundError(f"Could not load image: {self._query_image_path}")

        return image

    def _convert_to_gray(self, image: np.ndarray) -> np.ndarray:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def _find_sift_features(self, gray_image: np.ndarray) -> SiftFeatures | None:
        sift = cv.SIFT.create(nfeatures=self._sift_features_count)

        keypoints: Sequence[cv.KeyPoint]
        descriptors: np.ndarray | None

        keypoints, descriptors = sift.detectAndCompute(gray_image, None)

        if descriptors is None:
            return None

        return SiftFeatures(
            keypoints=tuple(keypoints),
            descriptors=descriptors,
        )

    def _find_good_matches(
        self,
        query_descriptors: np.ndarray,
        target_descriptors: np.ndarray,
    ) -> list[cv.DMatch]:
        bf_matcher = cv.BFMatcher(cv.NORM_L2)

        matches: Sequence[Sequence[cv.DMatch]] = bf_matcher.knnMatch(
            queryDescriptors=query_descriptors,
            trainDescriptors=target_descriptors,
            k=self._knn_matches_count,
        )

        good_matches: list[cv.DMatch] = []

        for match_pair in matches:
            if len(match_pair) < 2:
                continue

            best_match: cv.DMatch = match_pair[0]
            second_best_match: cv.DMatch = match_pair[1]

            if best_match.distance < self._lowe_ratio * second_best_match.distance:
                good_matches.append(best_match)

        return good_matches

    def _find_feature_match_result(
        self,
        query_image: np.ndarray,
        target_image: np.ndarray,
        query_features: SiftFeatures,
        target_features: SiftFeatures,
        last_good_corners: np.ndarray | None,
    ) -> FeatureMatchResult | None:
        good_matches: list[cv.DMatch] = self._find_good_matches(
            query_descriptors=query_features.descriptors,
            target_descriptors=target_features.descriptors,
        )

        if len(good_matches) < self._min_good_matches:
            return None

        query_points: np.ndarray = np.float32(
            [query_features.keypoints[match.queryIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)

        target_points: np.ndarray = np.float32(
            [target_features.keypoints[match.trainIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)

        homography: np.ndarray | None
        inlier_mask: np.ndarray | None

        homography, inlier_mask = cv.findHomography(
            srcPoints=query_points,
            dstPoints=target_points,
            method=cv.RANSAC,
            ransacReprojThreshold=self._ransac_reprojection_threshold,
        )

        if homography is None or inlier_mask is None:
            return None

        inlier_matches: list[cv.DMatch] = [
            match
            for match, is_inlier in zip(good_matches, inlier_mask.ravel(), strict=True)
            if int(is_inlier) == 1
        ]

        if len(inlier_matches) < self._min_inlier_matches:
            return None

        projected_corners: np.ndarray = self._project_query_corners(
            query_image=query_image,
            homography=homography,
        )

        if not self._is_outline_valid(
            target_image=target_image,
            projected_corners=projected_corners,
            last_good_corners=last_good_corners,
        ):
            return None

        if last_good_corners is not None:
            projected_corners = self._smooth_corners(
                previous_corners=last_good_corners,
                current_corners=projected_corners,
            )

        return FeatureMatchResult(
            inlier_matches=inlier_matches,
            projected_corners=projected_corners,
        )

    def _project_query_corners(
        self,
        query_image: np.ndarray,
        homography: np.ndarray,
    ) -> np.ndarray:
        query_height: int
        query_width: int
        query_height, query_width = query_image.shape[:2]

        query_corners: np.ndarray = np.float32(
            [
                [0, 0],
                [query_width, 0],
                [query_width, query_height],
                [0, query_height],
            ]
        ).reshape(-1, 1, 2)

        return cv.perspectiveTransform(
            src=query_corners,
            m=homography,
        )

    def _is_outline_valid(
        self,
        target_image: np.ndarray,
        projected_corners: np.ndarray,
        last_good_corners: np.ndarray | None,
    ) -> bool:
        target_height: int
        target_width: int
        target_height, target_width = target_image.shape[:2]

        contour: np.ndarray = projected_corners.reshape(-1, 2)

        if not np.all(np.isfinite(contour)):
            return False

        area: float = float(abs(cv.contourArea(contour)))
        target_area: float = float(target_width * target_height)

        min_area: float = target_area * self._min_outline_area_ratio
        max_area: float = target_area * self._max_outline_area_ratio

        if area < min_area or area > max_area:
            return False

        x_coordinates: np.ndarray = contour[:, 0]
        y_coordinates: np.ndarray = contour[:, 1]

        margin: int = 50

        if np.any(x_coordinates < -margin) or np.any(
            x_coordinates > target_width + margin
        ):
            return False

        if np.any(y_coordinates < -margin) or np.any(
            y_coordinates > target_height + margin
        ):
            return False

        if last_good_corners is not None:
            current_center: np.ndarray = np.mean(contour, axis=0)
            previous_center: np.ndarray = np.mean(
                last_good_corners.reshape(-1, 2),
                axis=0,
            )

            center_jump: float = float(np.linalg.norm(current_center - previous_center))

            if center_jump > self._max_center_jump_px:
                return False

        return True

    def _smooth_corners(
        self,
        previous_corners: np.ndarray,
        current_corners: np.ndarray,
    ) -> np.ndarray:
        return (
            self._smoothing_alpha * previous_corners
            + (1.0 - self._smoothing_alpha) * current_corners
        )

    def _draw_outline(
        self,
        target_image: np.ndarray,
        projected_corners: np.ndarray,
        is_last_good_position: bool,
    ) -> np.ndarray:
        result: np.ndarray = target_image.copy()

        color: tuple[int, int, int] = (
            self._last_position_color if is_last_good_position else self._outline_color
        )

        cv.polylines(
            img=result,
            pts=[np.int32(projected_corners)],
            isClosed=True,
            color=color,
            thickness=self._outline_thickness,
            lineType=cv.LINE_AA,
        )

        return result

    def _draw_matches(
        self,
        query_image: np.ndarray,
        query_features: SiftFeatures,
        target_image: np.ndarray,
        target_features: SiftFeatures,
        inlier_matches: list[cv.DMatch],
    ) -> np.ndarray:
        result: np.ndarray = cv.drawMatches(  # type: ignore[call-overload]
            img1=query_image,
            keypoints1=query_features.keypoints,
            img2=target_image,
            keypoints2=target_features.keypoints,
            matches1to2=inlier_matches,
            outImg=None,
            matchesThickness=1,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        return result

    def _compose_query_and_target(
        self,
        query_image: np.ndarray,
        target_image: np.ndarray,
    ) -> np.ndarray:
        query_height: int
        query_width: int
        query_height, query_width = query_image.shape[:2]

        target_height: int
        target_width: int
        target_height, target_width = target_image.shape[:2]

        output_height: int = max(query_height, target_height)
        output_width: int = query_width + target_width

        result: np.ndarray = np.zeros(
            (output_height, output_width, 3),
            dtype=np.uint8,
        )

        result[0:query_height, 0:query_width] = query_image
        result[0:target_height, query_width:output_width] = target_image

        return result

    def _draw_status(
        self,
        image: np.ndarray,
        text: str,
    ) -> None:
        cv.putText(
            img=image,
            text=text,
            org=(20, 35),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv.LINE_AA,
        )

    def run(self) -> None:
        query_image: np.ndarray = self._load_query_image()
        query_gray: np.ndarray = self._convert_to_gray(query_image)

        query_features: SiftFeatures | None = self._find_sift_features(query_gray)

        if query_features is None:
            raise ValueError("Could not compute SIFT features for query image.")

        last_good_corners: np.ndarray | None = None
        user_requested_exit: bool = False
        result_was_shown: bool = False

        try:
            while self._target_source.is_open():
                success: bool
                target_image: np.ndarray | None

                success, target_image = self._target_source.read()

                if not success or target_image is None:
                    break

                target_gray: np.ndarray = self._convert_to_gray(target_image)
                target_features: SiftFeatures | None = self._find_sift_features(
                    target_gray
                )

                match_result: FeatureMatchResult | None = None

                if target_features is not None:
                    match_result = self._find_feature_match_result(
                        query_image=query_image,
                        target_image=target_image,
                        query_features=query_features,
                        target_features=target_features,
                        last_good_corners=last_good_corners,
                    )

                if match_result is not None and target_features is not None:
                    target_with_outline: np.ndarray = self._draw_outline(
                        target_image=target_image,
                        projected_corners=match_result.projected_corners,
                        is_last_good_position=False,
                    )

                    result: np.ndarray = self._draw_matches(
                        query_image=query_image,
                        query_features=query_features,
                        target_image=target_with_outline,
                        target_features=target_features,
                        inlier_matches=match_result.inlier_matches,
                    )

                    last_good_corners = match_result.projected_corners

                    self._draw_status(
                        image=result,
                        text=f"TRACKING | INLIERS: {len(match_result.inlier_matches)}",
                    )

                elif last_good_corners is not None:
                    target_with_outline = self._draw_outline(
                        target_image=target_image,
                        projected_corners=last_good_corners,
                        is_last_good_position=True,
                    )

                    result = self._compose_query_and_target(
                        query_image=query_image,
                        target_image=target_with_outline,
                    )

                    self._draw_status(
                        image=result,
                        text="LAST GOOD POSITION",
                    )

                else:
                    result = self._compose_query_and_target(
                        query_image=query_image,
                        target_image=target_image,
                    )

                    self._draw_status(
                        image=result,
                        text="NO MATCH",
                    )

                cv.imshow("Feature matching", result)
                result_was_shown = True

                if cv.waitKey(1) & 0xFF == ord("q"):
                    user_requested_exit = True
                    break

            if result_was_shown and not user_requested_exit:
                cv.waitKey(0)

        finally:
            self._target_source.release()
            cv.destroyAllWindows()


if __name__ == "__main__":
    # target_source: FrameSource = ImageFileSource(
    #     Path("/Users/admin/Pycharm/MachineVision/project3/data/photo_2_train.jpg")
    # )
    #
    # tracker: FeatureMatchingTracker = FeatureMatchingTracker(
    #     query_image_path=Path(
    #         "/Users/admin/Pycharm/MachineVision/project3/data/photo_2_query.jpg"
    #     ),
    #     target_source=target_source,
    # )

    target_source: FrameSource = VideoFileSource(
        Path("/Users/admin/Pycharm/MachineVision/project3/data/video_3_train.mp4")
    )

    tracker: FeatureMatchingTracker = FeatureMatchingTracker(
        query_image_path=Path(
            "/Users/admin/Pycharm/MachineVision/project3/data/photo_3_query.jpg"
        ),
        target_source=target_source,
    )

    tracker.run()
