"""Track management primitives for droplets/spatters.

Provides lightweight Kalman-based tracking classes used by the tracking stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


def _build_kalman_filter(initial_pos):
    """Create a constant-velocity 2D Kalman filter initialized at a position."""
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]],
        dtype=np.float32,
    )
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

    x, y = float(initial_pos[0]), float(initial_pos[1])
    initial_state = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
    kalman.statePre = initial_state.copy()
    kalman.statePost = initial_state.copy()
    return kalman


@dataclass
class DropletObservation:
    """Single observation snapshot attached to a track."""
    frame_idx: int
    centroid: tuple[int, int]
    area: float | None
    diameter_px: float | None


class DropletTrack:
    """State holder for one tracked droplet/spatter object."""
    _next_id = 0

    def __init__(self, detection, frame_idx):
        """Start a new track from first detection."""
        self.id = DropletTrack._next_id
        DropletTrack._next_id += 1

        centroid = tuple(map(int, detection["centroid"]))
        self.kalman = _build_kalman_filter(centroid)
        self.predicted_position = centroid
        self.last_position = centroid
        self.hits = 1
        self.age = 1
        self.missed_frames = 0
        self.confirmed = False
        self.observations = [
            DropletObservation(
                frame_idx=frame_idx,
                centroid=centroid,
                area=detection.get("area"),
                diameter_px=detection.get("diameter_px"),
            )
        ]

    def predict(self):
        """Advance track state one step and return predicted centroid."""
        prediction = self.kalman.predict()
        self.age += 1
        self.predicted_position = (
            float(prediction[0, 0]),
            float(prediction[1, 0]),
        )
        return self.predicted_position

    def update(self, detection, frame_idx):
        """Correct filter state with a matched detection."""
        centroid = tuple(map(int, detection["centroid"]))
        measurement = np.array([[centroid[0]], [centroid[1]]], dtype=np.float32)
        self.kalman.correct(measurement)
        self.last_position = centroid
        self.missed_frames = 0
        self.hits += 1
        self.observations.append(
            DropletObservation(
                frame_idx=frame_idx,
                centroid=centroid,
                area=detection.get("area"),
                diameter_px=detection.get("diameter_px"),
            )
        )

    def mark_missed(self):
        """Mark this track as unmatched for the current frame."""
        self.missed_frames += 1

    def total_displacement(self):
        """Compute Euclidean displacement from first to latest observation."""
        if len(self.observations) < 2:
            return 0.0

        start_x, start_y = self.observations[0].centroid
        end_x, end_y = self.observations[-1].centroid
        return math.hypot(end_x - start_x, end_y - start_y)

    def vertical_displacement(self):
        """Compute signed vertical displacement from first to latest observation."""
        if len(self.observations) < 2:
            return 0.0
        return float(self.observations[-1].centroid[1] - self.observations[0].centroid[1])

    def latest_observation(self):
        """Return the most recent observation associated with this track."""
        return self.observations[-1]


class DropletTrackManager:
    """Manage active/finished tracks with matching and confirmation rules."""
    def __init__(
        self,
        max_match_distance=20.0,
        max_missed_frames=3,
        min_confirmed_hits=2,
        min_track_displacement=5.0,
        min_vertical_displacement=3.0,
    ):
        self.max_match_distance = max_match_distance
        self.max_missed_frames = max_missed_frames
        self.min_confirmed_hits = min_confirmed_hits
        self.min_track_displacement = min_track_displacement
        self.min_vertical_displacement = min_vertical_displacement
        self.active_tracks = []
        self.finished_tracks = []

    def _maybe_confirm(self, track):
        """Apply confirmation criteria and mark a track confirmed if eligible."""
        if track.confirmed:
            return True

        if track.hits < self.min_confirmed_hits:
            return False

        if track.total_displacement() < self.min_track_displacement:
            return False

        if track.vertical_displacement() < self.min_vertical_displacement:
            return False

        track.confirmed = True
        return True

    def _match_detections(self, detections, predicted_positions):
        """Greedily match detections to predicted track positions by distance."""
        candidate_pairs = []
        for track_idx, predicted in enumerate(predicted_positions):
            pred_x, pred_y = predicted
            for detection_idx, detection in enumerate(detections):
                det_x, det_y = detection["centroid"]
                distance = math.hypot(det_x - pred_x, det_y - pred_y)
                if distance <= self.max_match_distance:
                    candidate_pairs.append((distance, track_idx, detection_idx))

        candidate_pairs.sort(key=lambda item: item[0])

        matched_tracks = set()
        matched_detections = set()
        matches = []

        for _, track_idx, detection_idx in candidate_pairs:
            if track_idx in matched_tracks or detection_idx in matched_detections:
                continue
            matched_tracks.add(track_idx)
            matched_detections.add(detection_idx)
            matches.append((track_idx, detection_idx))

        unmatched_tracks = [idx for idx in range(len(self.active_tracks)) if idx not in matched_tracks]
        unmatched_detections = [idx for idx in range(len(detections)) if idx not in matched_detections]
        return matches, unmatched_tracks, unmatched_detections

    def update(self, detections, frame_idx):
        """Update tracking state for one frame and return confirmed tracks."""
        predicted_positions = [track.predict() for track in self.active_tracks]
        matches, unmatched_tracks, unmatched_detections = self._match_detections(detections, predicted_positions)

        confirmed_tracks_in_frame = []

        for track_idx, detection_idx in matches:
            track = self.active_tracks[track_idx]
            track.update(detections[detection_idx], frame_idx)
            if self._maybe_confirm(track):
                confirmed_tracks_in_frame.append(track)

        for track_idx in unmatched_tracks:
            self.active_tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            self.active_tracks.append(DropletTrack(detections[detection_idx], frame_idx))

        still_active_tracks = []
        for track in self.active_tracks:
            if track.missed_frames > self.max_missed_frames:
                self.finished_tracks.append(track)
            else:
                still_active_tracks.append(track)
        self.active_tracks = still_active_tracks

        return confirmed_tracks_in_frame

    def all_tracks(self):
        """Return finished and currently active tracks."""
        return [*self.finished_tracks, *self.active_tracks]
