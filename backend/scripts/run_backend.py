"""Native backend entry point, including psycopg-compatible Windows event loop."""

import asyncio

import uvicorn


def main() -> None:
    config = uvicorn.Config("app.main:app", host="127.0.0.1", port=8000, workers=1)
    asyncio.run(uvicorn.Server(config).serve(), loop_factory=asyncio.SelectorEventLoop)


if __name__ == "__main__":
    main()
