import cv2
import numpy as np

from taubenschreck.detector.sources.videofile import VideoFileSource


def _write_video(path, n_frames=5, w=64, h=48):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_videofile_yields_all_frames(tmp_path):
    vid = tmp_path / "clip.avi"
    _write_video(vid, n_frames=5)
    src = VideoFileSource(str(vid))
    frames = list(src.frames())
    src.close()
    assert len(frames) == 5
    assert frames[0].shape == (48, 64, 3)
