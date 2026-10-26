"""Turn a recording into text.

    python transcribe.py voice-note.m4a --language he
    python transcribe.py meeting.mp3 --timestamps

Any format the serving engine can read — WAV, MP3, M4A, OGG/Opus. One upload is at most the
pool's limit (24 MB unless the operator set otherwise); split longer recordings.
"""

import argparse

from gpm_client import PoolClient, PoolRequestError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording")
    parser.add_argument("--model", default="whisper-large-v3-turbo")
    parser.add_argument("--language", help="e.g. en, he — more accurate than detection when known")
    parser.add_argument("--prompt", help="names or terms the recording is likely to contain")
    parser.add_argument("--timestamps", action="store_true", help="print segments with their times")
    args = parser.parse_args()

    with PoolClient() as pool:
        try:
            transcript = pool.transcribe(
                args.model,
                args.recording,
                language=args.language,
                prompt=args.prompt,
                response_format="verbose_json" if args.timestamps else None,
            )
        except PoolRequestError as refused:
            # kind_not_served: that model does not transcribe.
            # request_too_large: split the recording.
            # generation_timeout: it ran past the pool's read time; split it.
            raise SystemExit(f"refused: {refused.reason} — {refused.detail}") from None

    if args.timestamps and transcript.segments:
        for segment in transcript.segments:
            print(f"[{segment['start']:7.2f} → {segment['end']:7.2f}] {segment['text'].strip()}")
    else:
        print(transcript.text.strip())
    print(f"\n(transcribed by {transcript.served_model} on {transcript.host})")


if __name__ == "__main__":
    main()
