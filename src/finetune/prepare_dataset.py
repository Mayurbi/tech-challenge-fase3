"""
Preparação do dataset de fine-tuning — curadoria, anonimização e formatação.
Responsável: Vinicius (Frente 1), com dados da Frente 2 (Natalia).

Entradas:
  - data/synthetic/protocolos_hospital.json  (protocolos, FAQ, laudos e recusas sintéticos)
  - data/raw/*.csv (opcional)                (Q&A externos, ex. MedQuAD/PubMedQA convertidos
                                              pela Frente 2 em colunas: question,answer[,source])

Saídas:
  - data/processed/train.jsonl   (formato de mensagens p/ SFT com chat template)
  - data/eval/perguntas_avaliacao.jsonl (perguntas retidas p/ comparação antes/depois)

Uso:
  python -m src.finetune.prepare_dataset [--raw-dir data/raw] [--max-external 500]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

from .prompts import SYSTEM_EXTERNAL_EN, format_chat

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYNTHETIC = os.path.join(ROOT, "data", "synthetic", "protocolos_hospital.json")
OUT_TRAIN = os.path.join(ROOT, "data", "processed", "train.jsonl")
OUT_EVAL = os.path.join(ROOT, "data", "eval", "perguntas_avaliacao.jsonl")

SEED = 42
N_EVAL_HOLDOUT = 6  # perguntas separadas para a avaliação antes/depois

# ---------------------------------------------------------------
# Anonimização (curadoria mínima obrigatória do edital)
# ---------------------------------------------------------------

_PATTERNS = [
    (re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"), "[CPF]"),
    (re.compile(r"\b\d{11}\b"), "[DOCUMENTO]"),
    (re.compile(r"\(?\d{2}\)?\s?9?\d{4}-?\d{4}\b"), "[TELEFONE]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"\bprontu[áa]rio\s+n?º?\s*\d+\b", re.IGNORECASE), "prontuário [Nº]"),
]


def anonymize(text: str) -> str:
    """Remove identificadores diretos (CPF, telefone, e-mail, nº de prontuário)."""
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


# ---------------------------------------------------------------
# Construção dos pares instrução/resposta
# ---------------------------------------------------------------

def _pairs_from_synthetic(data: dict) -> list[dict]:
    pairs: list[dict] = []

    for item in data.get("faq", []):
        pairs.append({"question": item["pergunta"], "answer": item["resposta"], "tag": "faq"})

    # Recusas: ensinam os limites de atuação (requisito de segurança)
    for item in data.get("recusas", []):
        pairs.append({"question": item["pergunta"], "answer": item["resposta"], "tag": "recusa"})

    # Protocolos: o modelo aprende a recitar o protocolo quando pedido
    for prot in data.get("protocolos", []):
        pairs.append({
            "question": f"Resuma o protocolo {prot['id']} ({prot['titulo']}).",
            "answer": f"{prot['conteudo']} Fonte: {prot['id']}.",
            "tag": "protocolo",
        })
        pairs.append({
            "question": f"O que diz o protocolo interno sobre {prot['titulo'].lower()}?",
            "answer": f"{prot['conteudo']} Fonte: {prot['id']}.",
            "tag": "protocolo",
        })

    for laudo in data.get("laudos_modelos", []):
        pairs.append({
            "question": f"Mostre o modelo institucional: {laudo['titulo'].lower()}.",
            "answer": f"{laudo['conteudo']}\nFonte: {laudo['id']}.",
            "tag": "laudo",
        })

    return pairs


def _pairs_from_raw_csv(raw_dir: str, limit: int) -> list[dict]:
    """Q&A externos preparados pela Frente 2 (colunas: question,answer[,source])."""
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        return []
    import pandas as pd  # import tardio: só é preciso se houver CSVs

    pairs: list[dict] = []
    for path in paths:
        df = pd.read_csv(path)
        cols = {c.lower().strip(): c for c in df.columns}
        if "question" not in cols or "answer" not in cols:
            print(f"[aviso] {os.path.basename(path)} sem colunas question/answer — ignorado")
            continue
        for _, row in df.iterrows():
            q, a = str(row[cols["question"]]).strip(), str(row[cols["answer"]]).strip()
            if len(q) < 8 or len(a) < 20:  # curadoria mínima
                continue
            src = str(row[cols["source"]]).strip() if "source" in cols else os.path.basename(path)
            pairs.append({"question": q, "answer": f"{a} Fonte: {src}.", "tag": "externo"})
    random.shuffle(pairs)
    return pairs[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara o dataset de fine-tuning")
    parser.add_argument("--raw-dir", default=os.path.join(ROOT, "data", "raw"))
    parser.add_argument("--max-external", type=int, default=500,
                        help="máx. de exemplos externos (MedQuAD etc.) a incluir")
    parser.add_argument("--synthetic-repeat", type=int, default=3,
                        help="repetições dos exemplos sintéticos PT quando há dados "
                             "externos, para o português não ser 'afogado' pelo inglês")
    args = parser.parse_args()

    random.seed(SEED)

    with open(SYNTHETIC, encoding="utf-8") as fh:
        synthetic = json.load(fh)

    pairs = _pairs_from_synthetic(synthetic)
    external = _pairs_from_raw_csv(args.raw_dir, args.max_external)
    print(f"Pares sintéticos: {len(pairs)} | externos: {len(external)}")

    # Holdout de avaliação: perguntas de FAQ retiradas do treino
    faq_idx = [i for i, p in enumerate(pairs) if p["tag"] == "faq"]
    eval_idx = set(random.sample(faq_idx, min(N_EVAL_HOLDOUT, len(faq_idx))))
    eval_pairs = [pairs[i] for i in sorted(eval_idx)]
    synthetic_train = [p for i, p in enumerate(pairs) if i not in eval_idx]
    # Com dados externos em inglês no treino, repetimos os exemplos PT para
    # manter o comportamento em português dominante.
    repeat = args.synthetic_repeat if external else 1
    train_pairs = synthetic_train * repeat + external
    random.shuffle(train_pairs)

    os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_EVAL), exist_ok=True)

    with open(OUT_TRAIN, "w", encoding="utf-8") as fh:
        for p in train_pairs:
            system = SYSTEM_EXTERNAL_EN if p["tag"] == "externo" else None
            messages = (format_chat(anonymize(p["question"]), anonymize(p["answer"]), system=system)
                        if system else
                        format_chat(anonymize(p["question"]), anonymize(p["answer"])))
            fh.write(json.dumps({"messages": messages, "tag": p["tag"]}, ensure_ascii=False) + "\n")

    with open(OUT_EVAL, "w", encoding="utf-8") as fh:
        for p in eval_pairs:
            fh.write(json.dumps({
                "question": anonymize(p["question"]),
                "reference": anonymize(p["answer"]),
            }, ensure_ascii=False) + "\n")

    print(f"OK: {len(train_pairs)} exemplos de treino -> {os.path.relpath(OUT_TRAIN, ROOT)}")
    print(f"OK: {len(eval_pairs)} perguntas de avaliação -> {os.path.relpath(OUT_EVAL, ROOT)}")


if __name__ == "__main__":
    main()
