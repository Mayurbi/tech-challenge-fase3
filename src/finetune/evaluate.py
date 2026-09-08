"""
Avaliação antes vs. depois do fine-tuning.
Responsável: Vinicius (Frente 1) — resultados vão para o relatório (Paola) e o vídeo (Thamy).

Roda as perguntas retidas em data/eval/perguntas_avaliacao.jsonl no modelo BASE e
no modelo FINE-TUNADO, mede ROUGE-L contra a resposta de referência e verifica se
a resposta cita a fonte (requisito de explainability). Gera:

  results/eval_before_after.jsonl  (par a par, para inspeção qualitativa)
  results/eval_summary.json        (métricas agregadas)

Uso (WSL2, GPU):
  python -m src.finetune.evaluate
"""

from __future__ import annotations

import json
import os
import re

from .prompts import SYSTEM_ASSISTENTE

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_FILE = os.path.join(ROOT, "data", "eval", "perguntas_avaliacao.jsonl")
OUT_DIR = os.path.join(ROOT, "results")

FONTE_RE = re.compile(r"\b(PROT-ONCO-\d+|LAUDO-[A-Z]+-\d+|REC-[A-Z]+-\d+|Fonte:)", re.IGNORECASE)


def _load_model(adapter_dir: str | None, cfg: dict):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir or cfg["model"]["base_model"],
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _generate(model, tokenizer, question: str, max_new_tokens: int = 320) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_ASSISTENTE},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens,
                         do_sample=False, repetition_penalty=1.3)
    text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    return text.strip()


def main() -> None:
    import yaml
    from rouge_score import rouge_scorer

    with open(os.path.join(ROOT, "configs", "finetune.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    adapter_dir = os.path.join(ROOT, cfg["output"]["adapter_dir"])
    if not os.path.isdir(adapter_dir):
        raise SystemExit(f"Adapter não encontrado em {adapter_dir}. Rode o treino antes.")

    with open(EVAL_FILE, encoding="utf-8") as fh:
        cases = [json.loads(line) for line in fh if line.strip()]
    print(f"{len(cases)} perguntas de avaliação")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    results, summary = [], {"base": {"rougeL": 0.0, "cita_fonte": 0},
                            "finetuned": {"rougeL": 0.0, "cita_fonte": 0}}

    for label, adapter in [("base", None), ("finetuned", adapter_dir)]:
        print(f"\n=== Gerando respostas: modelo {label} ===")
        model, tokenizer = _load_model(adapter, cfg)
        for case in cases:
            answer = _generate(model, tokenizer, case["question"])
            case[f"answer_{label}"] = answer
            score = scorer.score(case["reference"], answer)["rougeL"].fmeasure
            case[f"rougeL_{label}"] = round(score, 4)
            summary[label]["rougeL"] += score
            summary[label]["cita_fonte"] += bool(FONTE_RE.search(answer))
        del model  # libera VRAM antes de carregar o próximo
        import torch; torch.cuda.empty_cache()

    n = len(cases)
    for label in summary:
        summary[label]["rougeL"] = round(summary[label]["rougeL"] / n, 4)
        summary[label]["cita_fonte"] = f"{summary[label]['cita_fonte']}/{n}"

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "eval_before_after.jsonl"), "w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, "eval_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\n===== RESUMO =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDetalhe por pergunta: results/eval_before_after.jsonl")


if __name__ == "__main__":
    main()
