# Tech Challenge - Fase 3

## Assistente Virtual Médico com Fine-tuning de LLM e LangChain

**FIAP POS TECH - IADT (IA para Devs)**

Continuação das Fases 1 e 2: após a automação de análises de exames (modelos preditivos + SHAP)
e da interpretação via LLM, o hospital agora conta com um **assistente virtual médico** treinado
com os dados próprios da instituição, orquestrado com LangChain/LangGraph.

> ⚠️ Este sistema é uma ferramenta de **apoio à decisão clínica**. Ele **nunca prescreve
> diretamente, sem validação humana** — a decisão final é sempre do profissional de saúde.

## Equipe

| Integrante | Frente |
|---|---|
| Vinicius | Frente 1 — Fine-tuning da LLM (esta parte) |
| Thamy | Frente 2 — Dados (dataset de fine-tuning + base de prontuários) |
| Paola | Frente 3 — LangChain / LangGraph |
| Paola | Relatório técnico, diagrama, validação |
| Thamy | Vídeo e evidências |

## Frente 1 — Pipeline de Fine-tuning

- **Modelo base:** `unsloth/Llama-3.2-3B-Instruct` (4-bit)
- **Técnica:** QLoRA (LoRA sobre modelo quantizado em 4 bits) via [Unsloth](https://unsloth.ai)
- **Hardware alvo:** RTX 3070 (8 GB VRAM), rodando em **WSL2** — ver `docs/setup_wsl2.md`
- **Dados:** protocolos internos do hospital (sintéticos), FAQ de médicos e modelos de
  laudos/receitas, em formato instrução/resposta (`data/processed/*.jsonl`).
  A Frente 2 complementa com MedQuAD/PubMedQA convertidos pelo mesmo script.

### Estrutura

```
tech-challenge-fase3/
├── configs/finetune.yaml            # Hiperparâmetros do treino (edite aqui)
├── data/
│   ├── synthetic/                   # Protocolos/FAQ sintéticos do hospital (fonte)
│   ├── processed/                   # JSONL instrução/resposta gerado (entra no treino)
│   └── eval/                        # Perguntas de avaliação antes/depois
├── src/finetune/
│   ├── prompts.py                   # System prompt + guardrails (portado da Fase 2)
│   ├── prepare_dataset.py           # Curadoria/anonimização -> JSONL de treino
│   ├── train.py                     # Treino QLoRA com Unsloth
│   ├── evaluate.py                  # Comparação base vs. fine-tunado
│   └── inference.py                 # Chat local com o modelo + logging de auditoria
├── models/                          # Adapters LoRA salvos (gitignored)
├── logs/                            # Logs de auditoria (JSONL)
└── docs/setup_wsl2.md               # Setup do ambiente WSL2 + CUDA
```

### Como executar (dentro do WSL2)

```bash
# 1. Ambiente
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-finetune.txt

# 2. Gerar o dataset de treino a partir dos dados sintéticos
python -m src.finetune.prepare_dataset

# 3. Treinar (QLoRA na GPU)
python -m src.finetune.train

# 4. Avaliar: mesmas perguntas no modelo base e no fine-tunado
python -m src.finetune.evaluate

# 5. Conversar com o assistente (com guardrails e log de auditoria)
python -m src.finetune.inference --question "Qual o protocolo de triagem para nódulo mamário?"
python -m src.finetune.inference            # modo chat interativo
```

Os passos 2 (e os testes do 5 em modo `--offline`) rodam em qualquer máquina;
os passos 3–5 exigem GPU NVIDIA (WSL2 ou Colab).

### Saídas

- `models/lora-llama32-3b-medico/` — adapter LoRA (e opcionalmente o modelo mesclado)
- `results/eval_before_after.jsonl` — respostas base vs. fine-tunado por pergunta
- `logs/audit-YYYYMMDD.jsonl` — trilha de auditoria de cada interação

## Frente 2 — Dados (Natalia)

- **Dataset de fine-tuning**: além dos dados sintéticos do hospital, o
  `src/data/convert_medquad.py` converte o [MedQuAD](https://github.com/abachaa/MedQuAD)
  (subconjunto CancerGov, oncologia) para `data/raw/medquad_cancer.csv`, com curadoria de
  tamanho e fonte por par. O `prepare_dataset` mistura tudo: exemplos externos (em inglês)
  recebem um system prompt neutro em inglês, e os exemplos PT do hospital são repetidos
  (`--synthetic-repeat`, padrão 3x) para o português continuar dominante.
- **Base de prontuários (MySQL)**: `src/data/build_prontuarios.py` cria as tabelas
  `pacientes` (dados SEER anonimizados — sem raça/estado civil, id sintético),
  `exames` (pré-tratamento do PROT-ONCO-002, ~18% pendentes, sintéticos) e
  `alertas` (pendências >30 dias, PROT-ONCO-005) — é o banco que o LangChain consulta.

```bash
pip install -r requirements-data.txt

# MySQL: configure DATABASE_URL no .env (ver .env.example).
# Sem MySQL instalado: docker compose up -d
python -m src.data.build_prontuarios

# Dataset externo (opcional, já versionado em data/raw/medquad_cancer.csv):
git clone --depth 1 https://github.com/abachaa/MedQuAD.git /tmp/MedQuAD
python -m src.data.convert_medquad --medquad-dir /tmp/MedQuAD --limit 400
```

## Integração com as demais frentes

- A Frente 3 (LangChain) consome o modelo por carregamento direto do adapter
  (`FastLanguageModel.from_pretrained` + `PeftModel`) ou via **Ollama** após exportar GGUF
  (`python -m src.finetune.train --export-gguf`).
- O `prompts.SYSTEM_ASSISTENTE` é o mesmo usado no grafo LangGraph — fonte única de guardrails.

## Licença

MIT — Projeto acadêmico FIAP POS TECH 2026
