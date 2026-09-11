"""
Integração da Frente 3 com modelos locais executados pelo Ollama.

Responsável: Paola — Frente 3.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama

from src.finetune.prompts import SYSTEM_ASSISTENTE


class OllamaMedicalLLM:
    """
    Wrapper do modelo médico executado localmente pelo Ollama.

    O nome do modelo é configurável para permitir que o modelo
    base seja substituído posteriormente pelo modelo fine-tunado
    sem alterar o restante do LangChain ou LangGraph.
    """

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        temperature: float = 0.3,
        num_predict: int = 380,
    ) -> None:
        self.model_name = model_name

        self.client = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=num_predict,
            validate_model_on_init=True,
        )

    def invoke(
        self,
        prompt: str,
    ) -> str:
        """
        Envia o contexto clínico para o modelo local.
        """

        response = self.client.invoke(
            [
                (
                    "system",
                    SYSTEM_ASSISTENTE,
                ),
                (
                    "human",
                    prompt,
                ),
            ]
        )

        content = response.content

        if isinstance(content, str):
            return content.strip()

        return str(content).strip()