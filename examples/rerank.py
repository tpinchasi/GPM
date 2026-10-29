"""Rank documents against a query, best first.

    python rerank.py "How do I reset my password?" faq/*.txt

Each file is one document. A reranker reads the query and each document together, so it orders
them more accurately than comparing embeddings — use it on the few dozen candidates a search
already found, not on a whole collection.
"""

import argparse
from pathlib import Path

from gpm_client import PoolClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--model", default="bge-reranker-v2-m3")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    documents = [Path(name).read_text() for name in args.files]
    with PoolClient() as pool:
        ranked = pool.rerank(args.model, args.query, documents, top_n=args.top)

    for place, result in enumerate(ranked, 1):
        print(f"{place}. {result.score:.3f}  {args.files[result.index]}")
    print(f"\n(ranked by {ranked.served_model} on {ranked.host}, after {ranked.wait_s or 0:.1f}s in the queue)")


if __name__ == "__main__":
    main()
