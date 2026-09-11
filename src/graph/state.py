from __future__ import annotations

from typing import TypedDict


class MedicalAssistantState(
    TypedDict,
    total=False,
):
    """
    Estado trafegado entre os nós do LangGraph.
    """

    question: str
    paciente_id: str | None

    patient_context: str
    protocol_context: str

    sources: list[str]

    prompt: str
    answer: str

    model_name: str

    requires_human_review: bool
    risk_reasons: list[str]

    reviewer_name: str | None
    review_decision: str | None
    review_notes: str | None

    delivery_allowed: bool

    status: str