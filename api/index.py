import sys
from pathlib import Path

# Vercel mounts api/ as the function root — put the project root on sys.path
# so `app` and `seed` packages resolve.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402  (FastAPI ASGI app; Vercel expects the name `app`)
