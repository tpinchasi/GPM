"""The official OpenAI client against a pool, with the pool's waiting underneath.

    pip install openai 'gpm-client>=0.7'
    python openai_client.py voice-note.m4a receipt.jpg

The pool speaks OpenAI's /v1 paths, so the client works unchanged. Giving it the pool's
transport makes it wait for capacity, rather than fail, when every worker is busy.
"""

import base64
import mimetypes
import os
import sys

import httpx
from gpm_client import pool_transport
from openai import OpenAI

client = OpenAI(
    base_url=f"{os.environ['GPM_URL']}/v1",
    api_key=os.environ["GPM_API_KEY"],
    http_client=httpx.Client(transport=pool_transport(), timeout=httpx.Timeout(300, connect=5)),
)

recording, picture = sys.argv[1], sys.argv[2]

with open(recording, "rb") as audio:
    transcript = client.audio.transcriptions.create(model="whisper-large-v3-turbo", file=audio, language="he")
print("Transcript:", transcript.text)

mime = mimetypes.guess_type(picture)[0] or "image/jpeg"
with open(picture, "rb") as image:
    inline = f"data:{mime};base64,{base64.b64encode(image.read()).decode()}"   # links are refused
answer = client.chat.completions.create(model="gemma4:26b", messages=[{"role": "user", "content": [
    {"type": "text", "text": "What is in this picture?"},
    {"type": "image_url", "image_url": {"url": inline}},
]}])
print("Picture:", answer.choices[0].message.content)
