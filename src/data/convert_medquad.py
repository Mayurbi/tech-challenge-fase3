"""
Conversor MedQuAD -> CSV question/answer para o pipeline de fine-tuning.
Responsável: Natalia (Frente 2)

MedQuAD (https://github.com/abachaa/MedQuAD) é um conjunto de perguntas e
respostas médicas do NIH, em XML. Este script percorre os subconjuntos
escolhidos (por padrão CancerGov, todo em oncologia — mesmo domínio do
hospital), aplica curadoria (tamanho mínimo/máximo da resposta, remoção de
pares vazios) e gera data/raw/medquad_cancer.csv no formato lido pelo
src/finetune/prepare_dataset.py (colunas: question, answer, source).

Uso:
  git clone --depth 1 https://github.com/abachaa/MedQuAD.git /tmp/MedQuAD
  python -m src.data.convert_medquad --medquad-dir /tmp/MedQuAD --limit 400
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import re
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_CSV = os.path.join(ROOT, "data", "raw", "medquad_cancer.csv")

SUBSETS_DEFAULT = ["1_CancerGov_QA"]  # oncologia; adicione outros diretórios se quiser
MIN_ANSWER, MAX_ANSWER = 120, 1400    # curadoria: nem vazio, nem estourando o contexto

_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def _truncate(text: str, limit: int) -> str:
    """Trunca em limite de caracteres, sem cortar frase no meio."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    dot = cut.rfind(". ")
    return (cut[: dot + 1] if dot > limit * 0.5 else cut).strip()


def extract_pairs(medquad_dir: str, subsets: list[str]) -> list[dict]:
    pairs: list[dict] = []
    for subset in subsets:
        files = sorted(glob.glob(os.path.join(medquad_dir, subset, "*.xml")))
        for path in files:
            try:
                tree = ET.parse(path)
            except ET.ParseError:
                continue
            doc = tree.getroot()
            focus = _clean(doc.findtext("Focus") or "")
            source = doc.get("source") or subset
            for qa in doc.iter("QAPair"):
                question = _clean(qa.findtext("Question") or "")
                answer = _clean(qa.findtext("Answer") or "")
                if not question or len(answer) < MIN_ANSWER:
                    continue
                pairs.append({
                    "question": question,
                    "answer": _truncate(answer, MAX_ANSWER),
                    "source": f"MedQuAD/{source} ({focus})" if focus else f"MedQuAD/{source}",
                })
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Converte MedQuAD XML -> CSV")
    parser.add_argument("--medquad-dir", required=True, help="pasta do clone do MedQuAD")
    parser.add_argument("--subsets", nargs="*", default=SUBSETS_DEFAULT)
    parser.add_argument("--limit", type=int, default=400, help="máx. de pares no CSV")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = extract_pairs(args.medquad_dir, args.subsets)
    print(f"Pares extraídos após curadoria: {len(pairs)}")

    random.seed(args.seed)
    random.shuffle(pairs)
    pairs = pairs[: args.limit]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["question", "answer", "source"])
        writer.writeheader()
        writer.writerows(pairs)
    print(f"OK: {len(pairs)} pares -> {os.path.relpath(OUT_CSV, ROOT)}")


if __name__ == "__main__":
    main()
