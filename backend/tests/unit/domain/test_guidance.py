from __future__ import annotations

from app.domain.guidance import (
    GuidanceLevel,
)


def test_guidance_level_values_are_stable() -> None:
    assert GuidanceLevel.MINIMAL.value == "minimal"

    assert GuidanceLevel.GUIDED.value == "guided"

    assert GuidanceLevel.DIRECTED.value == "directed"

    assert GuidanceLevel.EXPLICIT.value == "explicit"


def test_guidance_level_can_be_constructed_from_string() -> None:
    assert GuidanceLevel("minimal") == GuidanceLevel.MINIMAL

    assert GuidanceLevel("guided") == GuidanceLevel.GUIDED

    assert GuidanceLevel("directed") == GuidanceLevel.DIRECTED

    assert GuidanceLevel("explicit") == GuidanceLevel.EXPLICIT


def test_guidance_level_has_four_levels() -> None:
    assert list(GuidanceLevel) == [
        GuidanceLevel.MINIMAL,
        GuidanceLevel.GUIDED,
        GuidanceLevel.DIRECTED,
        GuidanceLevel.EXPLICIT,
    ]
