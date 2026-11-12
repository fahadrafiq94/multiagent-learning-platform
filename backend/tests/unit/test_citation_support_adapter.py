import json

import pytest

from scripts.evaluate_citation_support import FIXTURE, agent_request, read_suite


def test_adapter_uses_frozen_synthetic_development_inputs_without_expectations_in_context():
    suite = read_suite(FIXTURE)
    assert len(suite["cases"]) == 8 and len(suite["generation_probes"]) == 3
    case = suite["cases"][0]
    request = agent_request(case["evidence"])
    assert request.context.grounding.items[1].content == case["evidence"][1]
    assert "expected" not in request.model_dump_json()
    assert case["answer"] not in request.user_message


@pytest.mark.parametrize(
    "field,value", [("split", "heldout"), ("synthetic", False), ("version", "other")]
)
def test_adapter_rejects_non_development_suites(tmp_path, field, value):
    data = read_suite(FIXTURE)
    data[field] = value
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic development"):
        read_suite(path)
