# Setup do ambiente de treino — WSL2 + RTX 3070

Guia para rodar o fine-tuning localmente no Windows com a RTX 3070 (8 GB).
O Unsloth roda em Linux; no Windows o caminho é o **WSL2**, que enxerga a GPU
através do driver NVIDIA do próprio Windows.

## 1. Instalar o WSL2 com Ubuntu

PowerShell **como administrador**:

```powershell
wsl --install -d Ubuntu-24.04
```

Reinicie se pedido, abra o "Ubuntu" no menu Iniciar e crie usuário/senha.
Se o WSL já estiver instalado, garanta a versão 2: `wsl --set-default-version 2`.

## 2. Driver NVIDIA (lado Windows)

Instale/atualize o **driver GeForce normal do Windows** (Game Ready ou Studio).
**Não** instale driver NVIDIA dentro do Ubuntu — o WSL2 usa o do Windows.

Verifique dentro do Ubuntu:

```bash
nvidia-smi   # deve listar a RTX 3070
```

## 3. Python e dependências (lado Ubuntu)

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git
cd ~ && git clone <URL_DO_REPO> tech-challenge-fase3 && cd tech-challenge-fase3
# (ou acesse o clone do Windows em /mnt/c/..., mas o disco Linux é bem mais rápido)

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-finetune.txt
```

O `unsloth` instala torch com CUDA, transformers, peft, trl e bitsandbytes
compatíveis entre si. Teste:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA GeForce RTX 3070
```

## 4. Acesso ao modelo base

O `unsloth/Llama-3.2-3B-Instruct` é baixado do Hugging Face na primeira execução
(~2,5 GB em 4-bit). Llama exige aceite de licença: crie conta no HF, aceite os
termos do modelo Meta Llama 3.2 e faça login uma vez:

```bash
pip install -U "huggingface_hub[cli]"
hf auth login   # cole um token de leitura
```

## 5. Rodar o pipeline

```bash
python -m src.finetune.prepare_dataset        # gera data/processed/train.jsonl
python -m src.finetune.train --max-steps 10   # smoke test (~2 min): confere VRAM/setup
python -m src.finetune.train                  # treino completo
python -m src.finetune.evaluate               # comparação base vs. fine-tunado
python -m src.finetune.inference              # chat com o assistente
```

## Dicas para os 8 GB da 3070

- Feche jogos/navegador com aceleração por GPU durante o treino (`nvidia-smi` mostra quem usa VRAM).
- Se der OOM: em `configs/finetune.yaml`, reduza `per_device_train_batch_size` para 1
  (compense subindo `gradient_accumulation_steps` para 16) ou `max_seq_length` para 768.
- O treino completo com o dataset sintético leva poucos minutos; com MedQuAD (500 ex.),
  espere algo em torno de 30–60 min por época.

## Plano B — Google Colab

Sem GPU disponível (ou para os colegas reproduzirem): rode os mesmos scripts num
notebook Colab com runtime T4. Basta `!pip install -r requirements-finetune.txt`,
subir o repo e chamar os mesmos módulos.
