"""
Inferência local do assistente com guardrails e log de auditoria.
Responsável: Vinicius (Frente 1) — a Frente 3 reutiliza este carregamento no LangChain.

Uso (WSL2, GPU):
  python -m src.finetune.inference --question "Qual o protocolo para BI-RADS 4?"
  python -m src.finetune.inference                 # chat interativo (sair: /q)
  python -m src.finetune.inference --base          # usa o modelo base (sem fine-tuning)

Cada interação é registrada em logs/audit-YYYYMMDD.jsonl com timestamp, pergunta,
resposta, modelo usado e fontes citadas — insumo direto do requisito de logging
e auditoria do edital (a Frente 3 estende esse formato para cada nó do grafo).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re

import yaml

from .prompts import DISCLAIMER, SYSTEM_ASSISTENTE

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(ROOT, "logs")
FONTE_RE = re.compile(r"\b(?:PROT|LAUDO|REC)-[A-Z]+-\d+\b")


def audit_log(event: dict) -> None:
    """Grava um evento na trilha de auditoria (JSONL, um arquivo por dia)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    event = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(), **event}
    path = os.path.join(LOG_DIR, f"audit-{dt.date.today():%Y%m%d}.jsonl")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_assistant(use_base: bool = False):
    """Carrega o modelo (base ou fine-tunado) pronto para inferência."""
    from unsloth import FastLanguageModel

    with open(os.path.join(ROOT, "configs", "finetune.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    adapter_dir = os.path.join(ROOT, cfg["output"]["adapter_dir"])
    name = cfg["model"]["base_model"] if use_base or not os.path.isdir(adapter_dir) else adapter_dir

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=name,
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer, name


def ask(model, tokenizer, model_name: str, question: str,
        history: list[dict] | None = None, max_new_tokens: int = 380) -> str:
    """Faz uma pergunta ao assistente, com guardrails, e registra auditoria."""
    messages = [{"role": "system", "content": SYSTEM_ASSISTENTE}]
    messages += history or []
    messages.append({"role": "user", "content": question})

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens,
                         temperature=0.3, top_p=0.9, do_sample=True, repetition_penalty=1.2)
    answer = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()

    audit_log({
        "etapa": "inferencia_llm",
        "modelo": model_name,
        "pergunta": question,
        "resposta": answer,
        "fontes_citadas": sorted(set(FONTE_RE.findall(answer))),
    })
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistente médico — inferência local")
    parser.add_argument("--question", "-q", help="pergunta única (senão, chat interativo)")
    parser.add_argument("--base", action="store_true", help="usar o modelo base, sem adapter")
    args = parser.parse_args()

    model, tokenizer, name = load_assistant(use_base=args.base)
    print(f"[modelo: {name}]")
    print(DISCLAIMER + "\n")

    if args.question:
        print(ask(model, tokenizer, name, args.question))
        return

    history: list[dict] = []
    print("Chat do assistente (sair: /q)")
    while True:
        try:
            question = input("\nmédico(a)> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("/q", "/quit", "sair"):
            break
        answer = ask(model, tokenizer, name, question, history)
        print(f"\nassistente> {answer}")
        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": answer}]
        history = history[-8:]  # janela curta para caber em 1024 tokens


if __name__ == "__main__":
    main()
