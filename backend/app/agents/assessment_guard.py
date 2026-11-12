"""Current-message provenance and consistency guard; no semantic keyword grading."""

from app.agents.schemas import StudentResponseAssessment, UnderstandingLevel


def guarded_assessment(
    assessment: StudentResponseAssessment | None, message: str
) -> tuple[StudentResponseAssessment | None, str]:
    if assessment is None:
        return None, "omitted"
    if assessment.assessability == "not_assessable":
        return assessment.model_copy(
            update={
                "understanding": UnderstandingLevel.NOT_APPLICABLE,
                "misconception_detected": False,
                "student_claim": None,
            }
        ), "not_assessable"
    if assessment.assessability != "assessable":
        return None, "uncertain"
    quote = assessment.student_claim
    if not quote or not quote.strip() or " ".join(quote.split()) not in " ".join(message.split()):
        return None, "unanchored"
    if assessment.understanding == UnderstandingLevel.NOT_APPLICABLE or (
        assessment.understanding == UnderstandingLevel.CORRECT and assessment.misconception_detected
    ):
        return None, "contradictory"
    return assessment, "accepted"
