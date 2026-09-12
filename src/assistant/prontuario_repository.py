from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.data.build_prontuarios import get_engine


class ProntuarioRepository:
    """
    Repositório de leitura da base de prontuários.

    Não permite SQL gerado livremente pela LLM.
    As consultas são pré-definidas e parametrizadas.
    Responsável: Paola
    """

    def __init__(
        self,
        engine: Engine | None = None,
    ) -> None:
        self.engine = engine or get_engine()

    def get_patient(
        self,
        paciente_id: str,
    ) -> dict[str, Any] | None:
        """
        Busca os dados clínicos de um paciente.
        """

        query = text(
            """
            SELECT
                paciente_id,
                idade,
                t_stage,
                n_stage,
                estagio_6th,
                diferenciacao,
                grau,
                a_stage,
                tamanho_tumor_mm,
                status_estrogeno,
                status_progesterona,
                linfonodos_examinados,
                linfonodos_positivos,
                meses_sobrevida,
                status
            FROM pacientes
            WHERE paciente_id = :paciente_id
            """
        )

        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    query,
                    {"paciente_id": paciente_id},
                )
                .mappings()
                .first()
            )

        if row is None:
            return None

        return dict(row)

    def get_exams(
        self,
        paciente_id: str,
    ) -> list[dict[str, Any]]:
        """
        Busca todos os exames registrados do paciente.
        """

        query = text(
            """
            SELECT
                exame,
                protocolo,
                status,
                data_solicitacao,
                data_resultado
            FROM exames
            WHERE paciente_id = :paciente_id
            ORDER BY data_solicitacao
            """
        )

        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    query,
                    {"paciente_id": paciente_id},
                )
                .mappings()
                .all()
            )

        return [
            dict(row)
            for row in rows
        ]

    def get_pending_exams(
        self,
        paciente_id: str,
    ) -> list[dict[str, Any]]:
        """
        Busca somente exames pendentes.
        """

        query = text(
            """
            SELECT
                exame,
                protocolo,
                status,
                data_solicitacao,
                data_resultado
            FROM exames
            WHERE paciente_id = :paciente_id
              AND status = 'pendente'
            ORDER BY data_solicitacao
            """
        )

        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    query,
                    {"paciente_id": paciente_id},
                )
                .mappings()
                .all()
            )

        return [
            dict(row)
            for row in rows
        ]

    def get_alerts(
        self,
        paciente_id: str,
    ) -> list[dict[str, Any]]:
        """
        Busca alertas clínicos existentes.
        """

        query = text(
            """
            SELECT
                tipo,
                prioridade,
                protocolo,
                detalhe,
                criado_em
            FROM alertas
            WHERE paciente_id = :paciente_id
            ORDER BY criado_em DESC
            """
        )

        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    query,
                    {"paciente_id": paciente_id},
                )
                .mappings()
                .all()
            )

        return [
            dict(row)
            for row in rows
        ]

    def build_context(
        self,
        paciente_id: str | None,
    ) -> str:
        """
        Transforma o prontuário em contexto textual seguro
        para a LLM.
        """

        if not paciente_id:
            return (
                "Nenhum paciente foi informado nesta consulta."
            )

        patient = self.get_patient(paciente_id)

        if patient is None:
            return (
                f"Paciente {paciente_id} não encontrado "
                "na base de prontuários."
            )

        exams = self.get_exams(paciente_id)
        alerts = self.get_alerts(paciente_id)

        lines: list[str] = [
            f"Paciente: {paciente_id}",
            f"Idade: {patient['idade']}",
            f"T Stage: {patient['t_stage']}",
            f"N Stage: {patient['n_stage']}",
            f"Estágio: {patient['estagio_6th']}",
            f"Grau: {patient['grau']}",
            (
                "Tamanho do tumor: "
                f"{patient['tamanho_tumor_mm']} mm"
            ),
            (
                "Status estrogênio: "
                f"{patient['status_estrogeno']}"
            ),
            (
                "Status progesterona: "
                f"{patient['status_progesterona']}"
            ),
            (
                "Linfonodos positivos: "
                f"{patient['linfonodos_positivos']}"
            ),
            "",
            "EXAMES:",
        ]

        if exams:
            for exam in exams:
                lines.append(
                    "- "
                    f"{exam['exame']} | "
                    f"status={exam['status']} | "
                    f"protocolo={exam['protocolo']} | "
                    f"solicitado={exam['data_solicitacao']} | "
                    f"resultado={exam['data_resultado']}"
                )
        else:
            lines.append(
                "- Nenhum exame registrado."
            )

        lines.append("")
        lines.append("ALERTAS:")

        if alerts:
            for alert in alerts:
                lines.append(
                    "- "
                    f"prioridade={alert['prioridade']} | "
                    f"tipo={alert['tipo']} | "
                    f"protocolo={alert['protocolo']} | "
                    f"{alert['detalhe']}"
                )
        else:
            lines.append(
                "- Nenhum alerta registrado."
            )

        return "\n".join(lines)