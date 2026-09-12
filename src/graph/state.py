from __future__ import annotations

from typing import Any,TypedDict


class MedicalAssistantState(
    TypedDict,
    total=False,
):
    """
    Estado trafegado entre os nós do LangGraph.
    Responsável: Paola
    """

    question: str
    paciente_id: str | None

    patient_context: str
    protocol_context: str
    verified_facts: dict[str, Any]

    response_consistent : bool
    consistency_issues: list[str]

    confidence_score: float
    confidence_level: str
    confidence_reasons: list[str]

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