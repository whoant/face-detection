import argparse
import json

from clip_generator import ClipGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate video clips using scene and speaker detection.")
    parser.add_argument('--video-path', required=True, help='Path to the local video file')
    parser.add_argument('--words-json', required=True, help='Path to JSON file containing words list')
    parser.add_argument('--output', required=False, help='Path to output JSON file (default: stdout)')
    parser.add_argument('--ffprobe-path', required=True, help='Path to ffprobe binary')
    parser.add_argument('--ffmpeg-path', required=True, help='Path to ffmpeg binary')
    args = parser.parse_args()
    print('Generating video clips...')
    # Load words from JSON file
    with open(args.words_json, 'r', encoding='utf-8') as f:
        words = json.load(f)

    # Run clip generation
    generator = ClipGenerator(words=words, video_path=args.video_path, ffprobe_path=args.ffprobe_path, ffmpeg_path=args.ffmpeg_path)
    clips = generator.process()

    # Output result
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({"clips": clips}, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps({"clips": clips}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()