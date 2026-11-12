import json

import pytest
from pydantic import ValidationError

from app.agents.scenario import onboarding_generation_schema
from app.agents.scenario_schemas import ScenarioDraftExtraction, ScenarioReviewDecision


@pytest.mark.parametrize("model", [ScenarioDraftExtraction, ScenarioReviewDecision])
def test_generation_schema_avoids_large_repeat_grammar_without_mutating_validation(model):
    schema = onboarding_generation_schema(model)
    assert "maxLength" not in json.dumps(schema)
    assert "maxLength" in json.dumps(model.model_json_schema())
    assert '"minLength": 1' in json.dumps(schema)
    assert schema["additionalProperties"] is False
    with pytest.raises(ValidationError):
        ScenarioDraftExtraction.model_validate({"company_context": "x" * 4001})
    with pytest.raises(ValidationError):
        ScenarioReviewDecision.model_validate({"intent": "confirm", "reason": "x" * 501})
