"""
Grounding clínico da Frente 3.

Responsável: Paola.

Extrai fatos objetivos do banco antes de enviar
o contexto para a LLM.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from src.assistant.prontuario_repository import ProntuarioRepository


def _json_safe_value(value: Any) -> Any:
    """
    Converte tipos do banco para valores
    compatíveis com JSON.
    """

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    return value


def _json_safe_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    """
    Converte todos os valores de um registro
    para tipos serializáveis em JSON.
    """

    return {
        key: _json_safe_value(value)
        for key, value in record.items()
    }


def build_verified_facts(
    repository: ProntuarioRepository,
    paciente_id: str | None,
) -> dict[str, Any]:
    """
    Obtém fatos verificados diretamente no banco.

    A LLM não participa desta etapa.
    """

    if not paciente_id:
        return {
            "patient_found": False,
            "pending_exams": [],
            "alerts": [],
            "related_protocols": [],
            "summary": (
                "Nenhum paciente foi informado. "
                "Não existem fatos clínicos verificados."
            ),
        }

    patient = repository.get_patient(
        paciente_id
    )

    if patient is None:
        return {
            "patient_found": False,
            "pending_exams": [],
            "alerts": [],
            "related_protocols": [],
            "summary": (
                f"Paciente {paciente_id} não encontrado "
                "na base de prontuários."
            ),
        }

    raw_pending_exams = (
        repository.get_pending_exams(
            paciente_id
        )
    )

    raw_alerts = repository.get_alerts(
        paciente_id
    )

    pending_exams = [
        _json_safe_record(exam)
        for exam in raw_pending_exams
    ]

    alerts = [
        _json_safe_record(alert)
        for alert in raw_alerts
    ]

    protocols: set[str] = set()

    for exam in pending_exams:
        protocol = exam.get(
            "protocolo"
        )

        if protocol:
            protocols.add(
                str(protocol)
            )

    for alert in alerts:
        protocol = alert.get(
            "protocolo"
        )

        if protocol:
            protocols.add(
                str(protocol)
            )

    lines: list[str] = [
        f"Paciente confirmado: {paciente_id}.",
        "",
        "EXAMES PENDENTES VERIFICADOS NO BANCO:",
    ]

    if pending_exams:
        for exam in pending_exams:
            lines.append(
                "- "
                f"{exam['exame']} | "
                f"solicitado={exam['data_solicitacao']} | "
                f"protocolo={exam['protocolo']}"
            )
    else:
        lines.append(
            "- Nenhum exame pendente."
        )

    lines.append("")
    lines.append(
        "ALERTAS VERIFICADOS NO BANCO:"
    )

    if alerts:
        for alert in alerts:
            lines.append(
                "- "
                f"tipo={alert['tipo']} | "
                f"prioridade={alert['prioridade']} | "
                f"protocolo={alert['protocolo']} | "
                f"detalhe={alert['detalhe']}"
            )
    else:
        lines.append(
            "- Nenhum alerta registrado."
        )

    lines.append("")
    lines.append(
        "REGRAS DE INTERPRETAÇÃO DOS FATOS:"
    )

    lines.append(
        "- Um alerta NÃO é um exame."
    )

    lines.append(
        "- Não inferir qual exame está pendente há mais "
        "de 30 dias se o alerta não identificar o exame."
    )

    lines.append(
        "- Não transformar exames realizados em exames pendentes."
    )

    lines.append(
        "- Não inventar resultados de exames."
    )

    return {
        "patient_found": True,
        "pending_exams": pending_exams,
        "alerts": alerts,
        "related_protocols": sorted(
            protocols
        ),
        "summary": "\n".join(
            lines
        ),
    }