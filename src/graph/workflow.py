from __future__ import annotations

import argparse
import json

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from src.graph.nodes import MedicalGraphNodes
from src.graph.state import MedicalAssistantState


class MedicalAssistantGraph:
    """
    Workflow principal da Frente 3.
    """

    def __init__(
        self,
        offline: bool = False,
        model_name: str = "llama3.2:3b",
        interactive_review: bool = False,
    ) -> None:
        self.nodes = MedicalGraphNodes(
            offline=offline,
            model_name=model_name,
            interactive_review=(
                interactive_review
            ),
        )

        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(
            MedicalAssistantState
        )

        builder.add_node(
            "validate_input",
            self.nodes.validate_input,
        )

        builder.add_node(
            "load_patient_context",
            self.nodes.load_patient_context,
        )

        builder.add_node(
            "retrieve_protocols",
            self.nodes.retrieve_protocols,
        )

        builder.add_node(
            "build_prompt",
            self.nodes.build_prompt,
        )

        builder.add_node(
            "generate_answer",
            self.nodes.generate_answer,
        )

        builder.add_node(
            "safety_check",
            self.nodes.safety_check,
        )

        builder.add_node(
            "human_review",
            self.nodes.human_review,
        )

        builder.add_node(
            "finalize",
            self.nodes.finalize,
        )

        builder.add_edge(
            START,
            "validate_input",
        )

        builder.add_edge(
            "validate_input",
            "load_patient_context",
        )

        builder.add_edge(
            "load_patient_context",
            "retrieve_protocols",
        )

        builder.add_edge(
            "retrieve_protocols",
            "build_prompt",
        )

        builder.add_edge(
            "build_prompt",
            "generate_answer",
        )

        builder.add_edge(
            "generate_answer",
            "safety_check",
        )

        builder.add_conditional_edges(
            "safety_check",
            self.nodes.route_after_safety,
            {
                "human_review": (
                    "human_review"
                ),
                "finalize": "finalize",
            },
        )

        builder.add_edge(
            "human_review",
            END,
        )

        builder.add_edge(
            "finalize",
            END,
        )

        return builder.compile()

    def invoke(
        self,
        question: str,
        paciente_id: str | None = None,
    ) -> MedicalAssistantState:
        result = self.graph.invoke(
            {
                "question": question,
                "paciente_id": paciente_id,
            }
        )

        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assistente médico — LangGraph"
        )
    )

    parser.add_argument(
        "--question",
        "-q",
        required=True,
        help="Pergunta clínica.",
    )

    parser.add_argument(
        "--paciente-id",
        "-p",
        default=None,
        help=(
            "ID sintético do paciente. "
            "Exemplo: P0001"
        ),
    )

    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Executa o workflow sem "
            "usar a LLM."
        ),
    )

    parser.add_argument(
        "--model",
        default="llama3.2:3b",
        help=(
            "Modelo disponível no Ollama. "
            "Padrão: llama3.2:3b"
        ),
    )

    parser.add_argument(
        "--interactive-review",
        action="store_true",
        help=(
            "Ativa a validação humana "
            "para respostas sensíveis."
        ),
    )

    args = parser.parse_args()

    assistant = MedicalAssistantGraph(
        offline=args.offline,
        model_name=args.model,
        interactive_review=(
            args.interactive_review
        ),
    )

    result = assistant.invoke(
        question=args.question,
        paciente_id=args.paciente_id,
    )

    print()
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()