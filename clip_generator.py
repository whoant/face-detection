from typing import Dict, List, Any, Optional

from active_speaker_detection import ActiveSpeakerDetection
from scene_algorithm import SceneAlgorithm


class ClipGenerator:
    def __init__(self, words: List[Dict[str, Any]], video_path: str, ffprobe_path: str, ffmpeg_path: str) -> None:
        self.words = words
        self.video_path = video_path
        self.ffprobe_path = ffprobe_path
        self.ffmpeg_path = ffmpeg_path

    def process(self) -> List[Dict[str, Any]]:
        """
        Process video to generate clips using scene detection and active speaker detection.
        """
        try:
            return self.generate_clips()
        except Exception as e:
            raise e

    def generate_clips(self) -> List[Dict[str, Any]]:
        def detect_scenes() -> List[Dict[str, Any]]:
            clips = []
            timeline_scenes = SceneAlgorithm.detect_scenes(self.video_path, threshold=0.1, ffprobe_path=self.ffprobe_path, ffmpeg_path=self.ffmpeg_path)  # type: ignore
            for i in range(len(timeline_scenes) - 1):
                start = timeline_scenes[i]
                end = timeline_scenes[i + 1]
                wordIds = [word['id'] for word in self.words if start <= word['start'] <= end]
                clips.append({
                    'start': start,
                    'end': end,
                    'wordIds': wordIds,
                })
            return clips

        def detect_active_speakers(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            detector = ActiveSpeakerDetection(words=self.words, clips=clips)
            return detector.detect(self.video_path)  # type: ignore

        clips = detect_scenes()
        return detect_active_speakers(clips)
