import re
import subprocess
import tempfile
import uuid
import os


def get_video_duration(ffprobe_path, video_path):
    command = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return float(result.stdout)


def gen_uuid():
    return str(uuid.uuid4())

class SceneAlgorithm:
    @staticmethod
    def detect_scenes(input_path='', threshold=0.5, ffprobe_path='', ffmpeg_path=''):
        output_path = f"{gen_uuid()}.txt"
        try:
            command = [
                ffmpeg_path,
                '-hide_banner',
                '-loglevel', 'error',
                '-i', input_path,
                '-filter_complex', f"select='gt(scene,{threshold})',metadata=print:file={output_path}",
                '-f', 'null', '-'
            ]
            subprocess.run(command)
            res = [0]
            pattern = r'pts_time:([\d.]+)'
            with open(output_path, mode='r') as f:
                lines = f.readlines()
                for line in lines:
                    match = re.search(pattern, line)
                    if match:
                        res.append(float(match.group(1)))
            res.append(get_video_duration(ffprobe_path, input_path))
            return res
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
