"""Export the backend OpenAPI contract for checked-in browser type generation."""

import argparse
import json
from pathlib import Path

from app.config.settings import Settings
from app.main import app

TARGET = Path(__file__).resolve().parents[2] / "frontend/generated/openapi.json"


def contract() -> str:
    schema = dict(app.openapi())
    # Deployment-specific title/version overrides do not change API types.
    schema["info"] = {
        **schema["info"],
        "title": Settings.model_fields["app_name"].default,
        "version": Settings.model_fields["app_version"].default,
    }
    return json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = contract()
    if args.check:
        if not TARGET.exists() or TARGET.read_text(encoding="utf-8") != expected:
            print("Frontend API contract is stale; regenerate it from the backend.")
            return 1
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
