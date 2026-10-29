# Examples

Small, complete programs against a GPM pool. Each reads the pool's address and your app key from
the environment and waits for capacity like every `gpm-client` call. The ones built on
`PoolClient` also print which build and host answered; through the OpenAI client those facts are
in each response's `X-GPM-*` headers.

```sh
pip install 'gpm-client>=0.7'        # 0.7 adds transcribe(), rerank() and image_part()
export GPM_URL=http://127.0.0.1:8090
export GPM_API_KEY=gpma_your_app_key  # an app key from the pool's operator — never an admin key
```

| File | What it shows |
|---|---|
| [`ask_about_a_picture.py`](ask_about_a_picture.py) | A question about a picture, sent inline in a chat message |
| [`transcribe.py`](transcribe.py) | A recording turned into text — with timestamps if the build gives them |
| [`rerank.py`](rerank.py) | Documents ranked against a query, best first |
| [`openai_client.py`](openai_client.py) | The same through the official OpenAI client, with the pool's waiting underneath and the client's own retries off |
| [`pool-beyond-chat.yaml`](pool-beyond-chat.yaml) | For operators: the catalog and hosts these examples assume |

```sh
python ask_about_a_picture.py receipt.jpg "What is the total?"
python transcribe.py voice-note.m4a --language he
python transcribe.py meeting.mp3 --timestamps
python rerank.py "How do I reset my password?" faq/*.txt
```

The model names are the ones in `pool-beyond-chat.yaml`; ask your operator for your pool's, or
pass `--model`. What each model can do, and what happens when it cannot, is in
[docs/models-beyond-chat.md](../docs/models-beyond-chat.md).
