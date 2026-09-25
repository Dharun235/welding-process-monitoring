"""AVI-to-frame extraction utility used by source discovery in the pipeline."""

from pathlib import Path

import cv2

def avi_to_frames(
    input_avi: Path,
    output_dir: Path,
    prefix: str = "frame_",
    start_index: int = 1,
    start_frame: int | None = None,
    end_frame: int | None = None,
    digits: int = 5,
    quality: int = 100,
):
    """
    Convert one AVI file to JPEG frames.

    Args:
    - start_frame: Zero-based source frame index to start from (inclusive).
    - end_frame: Zero-based source frame index to stop at (exclusive).

    Returns:
    - list[Path]: Saved frame file paths.
    """
    input_avi = Path(input_avi)
    output_dir = Path(output_dir)

    if input_avi.suffix.lower() != ".avi":
        raise ValueError(f"Expected an .avi file, got: {input_avi}")
    if not input_avi.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_avi}")
    if digits < 1:
        raise ValueError("digits must be >= 1")
    if quality < 0 or quality > 100:
        raise ValueError("quality must be between 0 and 100")
    if start_frame is not None and start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    if end_frame is not None and end_frame < 0:
        raise ValueError("end_frame must be >= 0")
    if start_frame is not None and end_frame is not None and end_frame < start_frame:
        raise ValueError("end_frame must be >= start_frame")

    cap = cv2.VideoCapture(str(input_avi))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_avi}")

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_index = start_index
    saved_frames = []
    current_source_frame = 0

    if start_frame is not None:
        seek_ok = cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        if seek_ok:
            current_source_frame = start_frame
        else:
            while current_source_frame < start_frame:
                success, _ = cap.read()
                if not success:
                    cap.release()
                    return saved_frames
                current_source_frame += 1

    while True:
        if end_frame is not None and current_source_frame >= end_frame:
            break

        success, frame = cap.read()
        if not success:
            break

        filename = f"{prefix}{frame_index:0{digits}d}.jpg"
        frame_path = output_dir / filename

        ok = cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to write frame: {frame_path}")

        saved_frames.append(frame_path)
        frame_index += 1
        current_source_frame += 1

    cap.release()
    return saved_frames


if __name__ == "__main__":
    print("Use avi_to_frames(input_avi, output_dir) from other scripts.")
