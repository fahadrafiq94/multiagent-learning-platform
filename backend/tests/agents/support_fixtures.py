"""Explicitly controlled judge output; this helper does not evaluate entailment."""

import json

from app.services.model_service.schemas import ModelChatResponse


def support_reply(request, *, verdict="supported", coverage="complete"):
    data = json.loads(request.messages[-1].content)
    return ModelChatResponse(
        content=json.dumps(
            {
                "answer_kind": "external_assertions" if data["claims"] else "non_assertive",
                "coverage": coverage,
                "teaching": "followed",
                "claims": [
                    {
                        "claim_index": index,
                        "verdict": verdict,
                        "quotes": [
                            {"label": label, "text": data["passages"][label][:600]}
                            for label in claim["citation_ids"]
                        ],
                    }
                    for index, claim in enumerate(data["claims"])
                ],
            }
        ),
        provider="ollama",
        model="controlled-judge",
        latency_ms=1,
    )
