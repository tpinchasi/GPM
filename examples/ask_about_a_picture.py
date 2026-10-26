"""Ask a question about a picture.

    python ask_about_a_picture.py receipt.jpg "What is the total?"

The picture travels inside the request (image_part reads and inlines it); the pool sends it only
to a build of the model that can see pictures, and refuses the request if none can.
"""

import argparse

from gpm_client import PoolClient, PoolRequestError, image_part


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("picture", help="a JPEG, PNG or WebP file")
    parser.add_argument("question")
    parser.add_argument("--model", default="gemma4:26b")
    args = parser.parse_args()

    with PoolClient() as pool:
        try:
            reply = pool.chat(args.model, [{"role": "user", "content": [
                {"type": "text", "text": args.question},
                image_part(args.picture),
            ]}])
        except PoolRequestError as refused:
            # input_not_accepted: no build of this model can see pictures.
            # request_too_large: the picture is over the pool's size limit.
            raise SystemExit(f"refused: {refused.reason} — {refused.detail}") from None

    print(reply.content)
    print(f"\n(answered by {reply.served_model} on {reply.host}, after {reply.wait_s or 0:.1f}s in the queue)")


if __name__ == "__main__":
    main()
