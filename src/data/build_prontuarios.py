"""
Base de prontuários do hospital (MySQL) — consumida pelo LangChain da Frente 3.
Responsável: Natalia (Frente 2)

Constrói as tabelas no MySQL a partir dos dados clínicos da Fase 1
(SEER, data/prontuarios/clinico_seer.csv):

  - pacientes:  dados clínicos ANONIMIZADOS (sem raça/estado civil; id sintético)
  - exames:     exames pré-tratamento (PROT-ONCO-002), parte deles pendente
  - alertas:    alertas de risco elevado (PROT-ONCO-005) já disparados

Os exames e alertas são SINTÉTICOS (gerados com seed fixa), apenas para
demonstrar os fluxos automatizados do assistente.

Conexão: variável de ambiente DATABASE_URL (ver .env.example). Padrão:
  mysql+pymysql://root:root@localhost:3306/hospital_mco
O banco (schema) é criado se não existir. Qualquer URL SQLAlchemy funciona —
ex.: sqlite:///data/prontuarios/prontuarios.db para testes sem MySQL.

Uso:
  python -m src.data.build_prontuarios
"""

from __future__ import annotations

import datetime as dt
import os
import random

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_IN = os.path.join(ROOT, "data", "prontuarios", "clinico_seer.csv")

DEFAULT_URL = "mysql+pymysql://root:root@localhost:3306/hospital_mco"

SEED = 42
EXAMES_PROT002 = [
    "Hemograma completo",
    "Função hepática (TGO/TGP, bilirrubinas)",
    "Função renal (creatinina, ureia)",
    "Ecocardiograma",
    "Estadiamento por imagem",
]


def _load_dotenv() -> None:
    """Carrega .env da raiz (sem dependências externas), como na Fase 2."""
    path = os.path.join(ROOT, ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def get_engine():
    """Engine SQLAlchemy apontando para DATABASE_URL; cria o schema MySQL se preciso."""
    _load_dotenv()
    url = make_url(os.environ.get("DATABASE_URL", DEFAULT_URL))

    if url.get_backend_name().startswith("mysql") and url.database:
        # criar o database se não existir, conectando sem schema
        # url.set(database=None) não limpa o campo — usar string vazia
        server = create_engine(url.set(database=""), pool_pre_ping=True)
        with server.connect() as conn:
            conn.execute(text(
                f"CREATE DATABASE IF NOT EXISTS `{url.database}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
            conn.commit()
        server.dispose()

    return create_engine(url, pool_pre_ping=True)


def build() -> None:
    random.seed(SEED)
    df = pd.read_csv(CSV_IN)
    df.columns = [c.strip() for c in df.columns]

    # --- pacientes: anonimização/curadoria -------------------------------
    # Removemos atributos demográficos sensíveis (raça, estado civil) e
    # criamos um id sintético — não há nomes/documentos na origem.
    pacientes = pd.DataFrame({
        "paciente_id": [f"P{idx + 1:04d}" for idx in range(len(df))],
        "idade": df["Age"],
        "t_stage": df["T Stage"].astype(str).str.strip(),
        "n_stage": df["N Stage"].astype(str).str.strip(),
        "estagio_6th": df["6th Stage"].astype(str).str.strip(),
        "diferenciacao": df["differentiate"].astype(str).str.strip(),
        "grau": df["Grade"].astype(str).str.strip(),
        "a_stage": df["A Stage"].astype(str).str.strip(),
        "tamanho_tumor_mm": df["Tumor Size"],
        "status_estrogeno": df["Estrogen Status"].astype(str).str.strip(),
        "status_progesterona": df["Progesterone Status"].astype(str).str.strip(),
        "linfonodos_examinados": df["Regional Node Examined"],
        "linfonodos_positivos": df["Reginol Node Positive"],
        "meses_sobrevida": df["Survival Months"],
        "status": df["Status"].astype(str).str.strip(),
    })

    hoje = dt.date.today()

    # --- exames: parte realizada, parte pendente (sintético) -------------
    exames_rows = []
    for pid in pacientes["paciente_id"]:
        for exame in EXAMES_PROT002:
            solicitado = hoje - dt.timedelta(days=random.randint(1, 60))
            pendente = random.random() < 0.18  # ~18% pendentes
            exames_rows.append({
                "paciente_id": pid,
                "exame": exame,
                "protocolo": "PROT-ONCO-002",
                "status": "pendente" if pendente else "realizado",
                "data_solicitacao": solicitado,
                "data_resultado": None if pendente else
                    solicitado + dt.timedelta(days=random.randint(1, 10)),
            })
    exames = pd.DataFrame(exames_rows)

    # --- alertas: pendência > 30 dias (PROT-ONCO-005) --------------------
    pend = exames[exames["status"] == "pendente"].copy()
    pend["dias"] = pend["data_solicitacao"].map(lambda d: (hoje - d).days)
    grp = pend[pend["dias"] > 30].groupby("paciente_id").size().reset_index(name="n")
    alertas = pd.DataFrame({
        "paciente_id": grp["paciente_id"],
        "tipo": "exames_pendentes_30d",
        "prioridade": "alta",
        "protocolo": "PROT-ONCO-005",
        "detalhe": grp["n"].map(
            lambda n: f"{n} exame(s) obrigatório(s) pendente(s) há mais de 30 dias"),
        "criado_em": hoje,
    })

    # --- gravar ----------------------------------------------------------
    engine = get_engine()
    with engine.begin() as conn:
        pacientes.to_sql("pacientes", conn, index=False, if_exists="replace")
        exames.to_sql("exames", conn, index=False, if_exists="replace")
        alertas.to_sql("alertas", conn, index=False, if_exists="replace")
        conn.execute(text("CREATE INDEX ix_exames_pac ON exames (paciente_id(10), status(10))")
                     if engine.url.get_backend_name().startswith("mysql")
                     else text("CREATE INDEX ix_exames_pac ON exames (paciente_id, status)"))

    print(f"OK: base gravada em {engine.url.render_as_string(hide_password=True)}")
    print(f"  pacientes: {len(pacientes)} | exames: {len(exames)} "
          f"(pendentes: {(exames['status'] == 'pendente').sum()}) | alertas: {len(alertas)}")

    # amostra de verificação (mesma consulta que o LangChain fará)
    with engine.connect() as conn:
        sample = conn.execute(text(
            "SELECT e.paciente_id, e.exame, e.data_solicitacao "
            "FROM exames e WHERE e.status = 'pendente' LIMIT 5")).fetchall()
    print("  exemplo de exames pendentes:", [tuple(map(str, r)) for r in sample])


if __name__ == "__main__":
    build()
