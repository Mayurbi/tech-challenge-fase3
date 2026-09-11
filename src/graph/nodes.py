from __future__ import annotations

import re
from typing import Any

from src.assistant.chain import CONTEXT_PROMPT
from src.assistant.ollama_llm import OllamaMedicalLLM
from src.assistant.prontuario_repository import (
    ProntuarioRepository,
)
from src.assistant.protocol_retriever import (
    ProtocolRetriever,
)
from src.finetune.inference import audit_log
from src.graph.state import MedicalAssistantState


class MedicalGraphNodes:
    """
    Implementação dos nós utilizados pelo LangGraph.
    """

    def __init__(
        self,
        offline: bool = False,
        model_name: str = "llama3.2:3b",
        interactive_review: bool = False,
    ) -> None:
        self.offline = offline
        self.interactive_review = (
            interactive_review
        )

        self.model_name = (
            "offline"
            if offline
            else model_name
        )

        self.protocol_retriever = (
            ProtocolRetriever(
                k=3,
            )
        )

        self.prontuario_repository = (
            ProntuarioRepository()
        )

        self.llm: OllamaMedicalLLM | None = None

        if not self.offline:
            self.llm = OllamaMedicalLLM(
                model_name=model_name,
            )

    def validate_input(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        question = str(
            state.get(
                "question",
                "",
            )
        ).strip()

        if not question:
            raise ValueError(
                "A pergunta clínica "
                "não pode ser vazia."
            )

        paciente_id = state.get(
            "paciente_id"
        )

        audit_log(
            {
                "etapa": (
                    "langgraph_validate_input"
                ),
                "paciente_id": paciente_id,
                "pergunta": question,
            }
        )

        return {
            "question": question,
            "paciente_id": paciente_id,
            "model_name": self.model_name,
            "status": "INPUT_VALIDATED",
        }

    def load_patient_context(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        paciente_id = state.get(
            "paciente_id"
        )

        patient_context = (
            self.prontuario_repository
            .build_context(
                paciente_id
            )
        )

        audit_log(
            {
                "etapa": (
                    "langgraph_patient_context"
                ),
                "paciente_id": paciente_id,
            }
        )

        return {
            "patient_context": (
                patient_context
            ),
            "status": "PATIENT_LOADED",
        }

    def retrieve_protocols(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        question = state["question"]

        documents = (
            self.protocol_retriever.search(
                question
            )
        )

        sources = sorted(
            {
                str(
                    document.metadata["id"]
                )
                for document in documents
            }
        )

        protocol_context = "\n\n".join(
            (
                f"[{document.metadata['id']}] "
                f"{document.metadata['titulo']}\n"
                f"{document.page_content}"
            )
            for document in documents
        )

        audit_log(
            {
                "etapa": (
                    "langgraph_protocol_retrieval"
                ),
                "paciente_id": state.get(
                    "paciente_id"
                ),
                "fontes_recuperadas": sources,
            }
        )

        return {
            "protocol_context": (
                protocol_context
            ),
            "sources": sources,
            "status": (
                "PROTOCOLS_RETRIEVED"
            ),
        }

    def build_prompt(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        sources = state.get(
            "sources",
            [],
        )

        prompt = CONTEXT_PROMPT.format(
            question=state["question"],
            patient_context=state.get(
                "patient_context",
                "Nenhum prontuário disponível.",
            ),
            protocol_context=state.get(
                "protocol_context",
                "Nenhum protocolo recuperado.",
            ),
            sources=(
                ", ".join(sources)
                if sources
                else "Nenhuma fonte recuperada"
            ),
        )

        audit_log(
            {
                "etapa": (
                    "langgraph_build_prompt"
                ),
                "paciente_id": state.get(
                    "paciente_id"
                ),
                "fontes": sources,
            }
        )

        return {
            "prompt": prompt,
            "status": "PROMPT_READY",
        }

    def generate_answer(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        if self.offline:
            sources = state.get(
                "sources",
                [],
            )

            answer = (
                "[MODO OFFLINE]\n"
                "Fluxo LangGraph executado até "
                "a etapa de geração.\n\n"
                f"Paciente: "
                f"{state.get('paciente_id') or 'não informado'}\n"
                "Protocolos recuperados: "
                f"{', '.join(sources) or 'nenhum'}\n\n"
                "A geração pela LLM está "
                "desativada neste teste."
            )

        else:
            if self.llm is None:
                raise RuntimeError(
                    "LLM não inicializada."
                )

            answer = self.llm.invoke(
                state["prompt"]
            )

        audit_log(
            {
                "etapa": (
                    "langgraph_generate_answer"
                ),
                "paciente_id": state.get(
                    "paciente_id"
                ),
                "modelo": self.model_name,
                "resposta": answer,
            }
        )

        return {
            "answer": answer,
            "model_name": self.model_name,
            "status": "ANSWER_GENERATED",
        }

    def safety_check(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        """
        Detecta pedidos clínicos sensíveis.

        Perguntas relacionadas a prescrição,
        dosagem, mudança de medicamento ou
        diagnóstico definitivo exigem revisão.

        Também detecta doses concretas geradas
        pelo modelo.
        """

        question = state.get(
            "question",
            "",
        ).lower()

        answer = state.get(
            "answer",
            "",
        ).lower()

        question_rules = {
            "prescricao": (
                r"\b("
                r"prescrev\w*|"
                r"prescriç\w*|"
                r"prescric\w*"
                r")\b"
            ),
            "dosagem": (
                r"\b("
                r"dose|"
                r"doses|"
                r"dosagem"
                r")\b"
            ),
            "alteracao_medicamento": (
                r"\b("
                r"suspender|"
                r"interromper|"
                r"trocar medicamento|"
                r"alterar medicamento"
                r")\b"
            ),
            "diagnostico_direto": (
                r"\b("
                r"diagnostique|"
                r"diagnóstico definitivo|"
                r"diagnostico definitivo"
                r")\b"
            ),
        }

        reasons: list[str] = []

        for reason, pattern in (
            question_rules.items()
        ):
            if re.search(
                pattern,
                question,
                flags=re.IGNORECASE,
            ):
                reasons.append(
                    reason
                )

        concrete_dose_pattern = (
            r"\b\d+(?:[.,]\d+)?\s*"
            r"(?:mg|ml|mcg|µg)\b"
        )

        if re.search(
            concrete_dose_pattern,
            answer,
            flags=re.IGNORECASE,
        ):
            reasons.append(
                "dose_concreta_gerada"
            )

        reasons = list(
            dict.fromkeys(
                reasons
            )
        )

        requires_human_review = bool(
            reasons
        )

        audit_log(
            {
                "etapa": (
                    "langgraph_safety_check"
                ),
                "paciente_id": state.get(
                    "paciente_id"
                ),
                "requires_human_review": (
                    requires_human_review
                ),
                "risk_reasons": reasons,
            }
        )

        return {
            "requires_human_review": (
                requires_human_review
            ),
            "risk_reasons": reasons,
            "status": "SAFETY_CHECKED",
        }

    def human_review(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        if not self.interactive_review:
            audit_log(
                {
                    "etapa": (
                        "langgraph_human_review_required"
                    ),
                    "paciente_id": state.get(
                        "paciente_id"
                    ),
                    "motivos": state.get(
                        "risk_reasons",
                        [],
                    ),
                    "status": (
                        "PENDING_HUMAN_REVIEW"
                    ),
                }
            )

            return {
                "reviewer_name": None,
                "review_decision": None,
                "review_notes": None,
                "delivery_allowed": False,
                "status": (
                    "PENDING_HUMAN_REVIEW"
                ),
            }

        print()
        print("=" * 60)
        print(
            "REVISÃO HUMANA OBRIGATÓRIA"
        )
        print("=" * 60)

        print(
            f"Paciente: "
            f"{state.get('paciente_id') or 'não informado'}"
        )

        print(
            "Motivos: "
            f"{', '.join(state.get('risk_reasons', []))}"
        )

        print()
        print("Pergunta:")
        print(
            state.get(
                "question",
                "",
            )
        )

        print()
        print("Resposta gerada:")
        print(
            state.get(
                "answer",
                "",
            )
        )

        print()
        print("Fontes:")
        print(
            ", ".join(
                state.get(
                    "sources",
                    [],
                )
            )
            or "Nenhuma"
        )

        print("=" * 60)

        reviewer_name = input(
            "Nome do profissional responsável: "
        ).strip()

        while not reviewer_name:
            print(
                "O nome do profissional "
                "é obrigatório."
            )

            reviewer_name = input(
                "Nome do profissional responsável: "
            ).strip()

        decision_input = input(
            "Decisão [A]provar / [R]ejeitar: "
        ).strip().lower()

        while decision_input not in {
            "a",
            "aprovar",
            "aprovado",
            "r",
            "rejeitar",
            "rejeitado",
        }:
            print(
                "Informe A para aprovar "
                "ou R para rejeitar."
            )

            decision_input = input(
                "Decisão [A]provar / [R]ejeitar: "
            ).strip().lower()

        review_notes = input(
            "Observações da revisão: "
        ).strip()

        approved = decision_input in {
            "a",
            "aprovar",
            "aprovado",
        }

        review_decision = (
            "APPROVED"
            if approved
            else "REJECTED"
        )

        status = (
            "HUMAN_APPROVED"
            if approved
            else "HUMAN_REJECTED"
        )

        audit_log(
            {
                "etapa": (
                    "langgraph_human_review_completed"
                ),
                "paciente_id": state.get(
                    "paciente_id"
                ),
                "profissional": (
                    reviewer_name
                ),
                "decisao": (
                    review_decision
                ),
                "observacoes": (
                    review_notes
                ),
                "motivos_risco": state.get(
                    "risk_reasons",
                    [],
                ),
                "fontes": state.get(
                    "sources",
                    [],
                ),
                "resposta_validada": (
                    state.get(
                        "answer",
                        "",
                    )
                ),
            }
        )

        return {
            "reviewer_name": (
                reviewer_name
            ),
            "review_decision": (
                review_decision
            ),
            "review_notes": (
                review_notes
            ),
            "delivery_allowed": approved,
            "status": status,
        }

    def finalize(
        self,
        state: MedicalAssistantState,
    ) -> dict[str, Any]:
        audit_log(
            {
                "etapa": (
                    "langgraph_completed"
                ),
                "paciente_id": state.get(
                    "paciente_id"
                ),
                "modelo": self.model_name,
                "fontes": state.get(
                    "sources",
                    [],
                ),
                "delivery_allowed": True,
            }
        )

        return {
            "delivery_allowed": True,
            "status": "COMPLETED",
        }

    @staticmethod
    def route_after_safety(
        state: MedicalAssistantState,
    ) -> str:
        if state.get(
            "requires_human_review",
            False,
        ):
            return "human_review"

        return "finalize"