from __future__ import annotations

import argparse
import json
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from src.assistant.ollama_llm import OllamaMedicalLLM
from src.assistant.prontuario_repository import (
    ProntuarioRepository,
)
from src.assistant.protocol_retriever import (
    ProtocolRetriever,
)
from src.finetune.inference import audit_log


CONTEXT_PROMPT = PromptTemplate.from_template(
    """
Responda à pergunta clínica abaixo utilizando SOMENTE as
informações do prontuário e os protocolos internos recuperados.

Não invente dados ausentes.

Caso a informação não seja suficiente, informe explicitamente
que não há dados suficientes.

Toda decisão clínica final depende da validação do médico responsável.

PERGUNTA:
{question}

DADOS DO PRONTUÁRIO:
{patient_context}

PROTOCOLOS INTERNOS RECUPERADOS:
{protocol_context}

FONTES DISPONÍVEIS:
{sources}

Responda em português do Brasil.

Utilize apenas as informações presentes no contexto fornecido.

Cite explicitamente os códigos dos protocolos utilizados,
como PROT-ONCO-001.

Não forneça diagnóstico definitivo ou prescrição médica autônoma.
""".strip()
)


class MedicalAssistantChain:
    """
    Pipeline LangChain principal da Frente 3.
    """

    def __init__(
        self,
        offline: bool = False,
        model_name: str = "llama3.2:3b",
    ) -> None:
        self.offline = offline
        self.model_name = (
            "offline"
            if offline
            else model_name
        )

        self.protocol_retriever = ProtocolRetriever(
            k=3,
        )

        self.prontuario_repository = (
            ProntuarioRepository()
        )

        self.llm: OllamaMedicalLLM | None = None

        if not self.offline:
            self.llm = OllamaMedicalLLM(
                model_name=model_name,
            )

        self.chain = (
            RunnableLambda(
                self._retrieve_context
            )
            | RunnableLambda(
                self._generate_answer
            )
        )

    def _retrieve_context(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Etapa de retrieval do LangChain.
        """

        question = str(
            payload.get("question", "")
        ).strip()

        paciente_id = payload.get(
            "paciente_id"
        )

        if not question:
            raise ValueError(
                "A pergunta não pode ser vazia."
            )

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

        patient_context = (
            self.prontuario_repository
            .build_context(
                paciente_id
            )
        )

        result = {
            "question": question,
            "paciente_id": paciente_id,
            "patient_context": patient_context,
            "protocol_context": protocol_context,
            "sources": sources,
            "model_name": self.model_name,
        }

        audit_log(
            {
                "etapa": (
                    "langchain_retrieval"
                ),
                "paciente_id": paciente_id,
                "pergunta": question,
                "fontes_recuperadas": sources,
            }
        )

        return result

    def _generate_answer(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Gera a resposta final usando o
        contexto recuperado.
        """

        prompt = CONTEXT_PROMPT.format(
            question=context["question"],
            patient_context=(
                context[
                    "patient_context"
                ]
            ),
            protocol_context=(
                context[
                    "protocol_context"
                ]
            ),
            sources=(
                ", ".join(
                    context["sources"]
                )
                or "Nenhuma fonte recuperada"
            ),
        )

        if self.offline:
            answer = (
                "[MODO OFFLINE]\n"
                "Pipeline LangChain executado "
                "com sucesso.\n\n"
                f"Paciente: "
                f"{context['paciente_id'] or 'não informado'}\n"
                "Fontes recuperadas: "
                f"{', '.join(context['sources']) or 'nenhuma'}\n\n"
                "A geração da resposta pela LLM "
                "foi desativada neste teste."
            )

        else:
            if self.llm is None:
                raise RuntimeError(
                    "LLM não inicializada."
                )

            answer = self.llm.invoke(
                prompt
            )

        audit_log(
            {
                "etapa": (
                    "langchain_resposta"
                ),
                "paciente_id": (
                    context["paciente_id"]
                ),
                "modelo": self.model_name,
                "fontes_recuperadas": (
                    context["sources"]
                ),
                "resposta": answer,
            }
        )

        return {
            **context,
            "prompt": prompt,
            "answer": answer,
        }

    def invoke(
        self,
        question: str,
        paciente_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Executa o pipeline LangChain.
        """

        return self.chain.invoke(
            {
                "question": question,
                "paciente_id": paciente_id,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assistente médico — LangChain"
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
            "Testa retrieval e banco "
            "sem executar a LLM."
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

    args = parser.parse_args()

    assistant = MedicalAssistantChain(
        offline=args.offline,
        model_name=args.model,
    )

    result = assistant.invoke(
        question=args.question,
        paciente_id=args.paciente_id,
    )

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