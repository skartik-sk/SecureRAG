import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.db import init_db  # noqa: E402
from seed import seed_all  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Seed demo workspaces")
    parser.add_argument("--corpus", default="my_docs_folder")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    if not settings.groq_api_key:
        raise SystemExit("GROQ_API_KEY missing in .env")
    init_db(settings)

    counts = seed_all(args.corpus, settings, notify=print)
    print("\nSeeded chunk totals:")
    for slug, n in counts.items():
        print(f"  {slug}: {n}")


if __name__ == "__main__":
    main()
