from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROTOCOL_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "protocolos_hospital.json"
)


def _normalize_tokens(text: str) -> list[str]:
    """
    Normaliza texto para melhorar a busca BM25.

    Remove acentos, converte para minúsculas e separa
    o conteúdo em tokens.
    """

    normalized = unicodedata.normalize("NFKD", text.lower())

    normalized = "".join(
        char
        for char in normalized
        if not unicodedata.combining(char)
    )

    return re.findall(
        r"[a-z0-9-]+",
        normalized,
    )


class ProtocolRetriever:
    """
    Recuperador de protocolos internos do hospital.

    Utiliza BM25Retriever do LangChain para localizar protocolos
    relevantes sem depender de API externa ou banco vetorial.
    """

    def __init__(
        self,
        protocol_path: Path | str = DEFAULT_PROTOCOL_PATH,
        k: int = 3,
    ) -> None:
        self.protocol_path = Path(protocol_path)

        if not self.protocol_path.exists():
            raise FileNotFoundError(
                f"Arquivo de protocolos não encontrado: "
                f"{self.protocol_path}"
            )

        self.documents = self._load_documents()

        if not self.documents:
            raise ValueError(
                "Nenhum protocolo foi encontrado no arquivo."
            )

        self.retriever = BM25Retriever.from_documents(
            self.documents,
            preprocess_func=_normalize_tokens,
        )

        self.retriever.k = k

    def _load_documents(self) -> list[Document]:
        """
        Carrega protocolos e modelos do JSON como Documents
        do LangChain.
        """

        with self.protocol_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        documents: list[Document] = []

        sections = (
            "protocolos",
            "laudos_modelos",
        )

        for section in sections:
            for item in data.get(section, []):
                document = Document(
                    page_content=(
                        f"{item['titulo']}\n\n"
                        f"{item['conteudo']}"
                    ),
                    metadata={
                        "id": item["id"],
                        "titulo": item["titulo"],
                        "tipo": section,
                        "fonte": str(self.protocol_path),
                    },
                )

                documents.append(document)

        return documents

    def search(
        self,
        question: str,
    ) -> list[Document]:
        """
        Recupera os protocolos mais relevantes para uma pergunta.
        """

        question = question.strip()

        if not question:
            return []

        return self.retriever.invoke(question)


if __name__ == "__main__":
    retriever = ProtocolRetriever()

    pergunta = (
        "Quais exames são necessários antes do tratamento?"
    )

    resultados = retriever.search(pergunta)

    print("\nProtocolos encontrados:\n")

    for document in resultados:
        print(
            f"{document.metadata['id']} - "
            f"{document.metadata['titulo']}"
        )