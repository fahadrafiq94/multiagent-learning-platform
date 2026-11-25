"""Serve only browser assets, under the same origin as authenticated APIs."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2] / "frontend"
HEADERS = {
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
}


def configure_student_frontend(app: FastAPI) -> None:
    @app.get("/student", include_in_schema=False)
    @app.get("/student/", include_in_schema=False)
    def student_entry():
        path = ROOT / "student.html"
        if not path.is_file():
            raise HTTPException(404, detail="Student frontend is not installed.")
        return FileResponse(path, headers=HEADERS)

    for directory in ("css", "js"):
        path = ROOT / directory
        if path.is_dir():
            app.mount(
                f"/student/{directory}", StaticFiles(directory=path), name=f"student-{directory}"
            )
