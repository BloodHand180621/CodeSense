"""Structured feedback intake built on the existing system log table.

The first feedback-center slice deliberately avoids a schema migration.  A
versioned JSON record gives the support workflow a stable contract today;
later status transitions can migrate the same fields into a dedicated table
without changing the public form contract.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

from models import SystemLog, db


FEEDBACK_SCHEMA_VERSION = 1
FEEDBACK_STATUS_RECEIVED = "received"
FEEDBACK_CATEGORIES = (
    ("bug", "功能异常"),
    ("content", "内容或评测"),
    ("experience", "使用体验"),
    ("privacy", "隐私与数据"),
    ("other", "其他"),
)
FEEDBACK_CATEGORY_LABELS = dict(FEEDBACK_CATEGORIES)
FEEDBACK_ID_PATTERN = re.compile(r"^FB-[0-9A-F]{12}$")
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class FeedbackValidationError(ValueError):
    """Raised when a feedback submission cannot be accepted safely."""

    def __init__(self, errors: dict[str, str]):
        super().__init__("feedback validation failed")
        self.errors = errors


def _clean(value, *, max_length: int) -> str:
    """Normalize a user-provided field without changing its meaning."""

    if value is None:
        return ""
    return str(value).strip()[:max_length]


def _text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_feedback_payload(data) -> dict[str, str]:
    """Validate and normalize the public feedback form fields."""

    data = data or {}
    category = _text(data.get("category"))
    subject = _text(data.get("subject"))
    message = _text(data.get("message"))
    reproduction_steps = _text(data.get("reproduction_steps"))
    page_context = _text(data.get("page_context"))
    contact_email = _text(data.get("contact_email"))

    errors = {}
    if category not in FEEDBACK_CATEGORY_LABELS:
        errors["category"] = "请选择一个反馈类型。"
    if len(subject) < 3:
        errors["subject"] = "主题至少需要 3 个字符。"
    elif len(subject) > 120:
        errors["subject"] = "主题不能超过 120 个字符。"
    if len(message) < 10:
        errors["message"] = "请提供至少 10 个字符的具体描述。"
    elif len(message) > 4000:
        errors["message"] = "具体描述不能超过 4000 个字符。"
    if len(reproduction_steps) > 2000:
        errors["reproduction_steps"] = "复现步骤不能超过 2000 个字符。"
    if len(page_context) > 200:
        errors["page_context"] = "页面或请求上下文不能超过 200 个字符。"
    if len(contact_email) > 120:
        errors["contact_email"] = "联系邮箱不能超过 120 个字符。"
    elif contact_email and not _EMAIL_PATTERN.fullmatch(contact_email):
        errors["contact_email"] = "请输入有效的电子邮箱，或留空。"

    if errors:
        raise FeedbackValidationError(errors)

    return {
        "category": category,
        "subject": subject,
        "message": message,
        "reproduction_steps": reproduction_steps,
        "page_context": page_context or "/feedback",
        "contact_email": contact_email,
    }


def create_feedback_record(
    data,
    *,
    request_context: dict[str, str | None],
) -> dict:
    """Create a versioned record with opaque request correlation fields."""

    payload = normalize_feedback_payload(data)
    submitted_at = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    feedback_id = f"FB-{uuid.uuid4().hex[:12].upper()}"
    return {
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "feedback_id": feedback_id,
        "status": FEEDBACK_STATUS_RECEIVED,
        "status_history": [
            {"status": FEEDBACK_STATUS_RECEIVED, "at": submitted_at}
        ],
        "submitted_at": submitted_at,
        "category": payload["category"],
        "category_label": FEEDBACK_CATEGORY_LABELS[payload["category"]],
        "subject": payload["subject"],
        "message": payload["message"],
        "reproduction_steps": payload["reproduction_steps"],
        "contact_email": payload["contact_email"],
        "context": {
            "page": payload["page_context"],
            "request_id": _clean(request_context.get("request_id"), max_length=64),
            "endpoint": _clean(request_context.get("endpoint"), max_length=120),
            "method": _clean(request_context.get("method"), max_length=12),
        },
    }


def save_feedback(record: dict, *, user_id: str | None = None) -> dict:
    """Persist one feedback submission as an existing system-log event."""

    log = SystemLog(
        log_type="反馈提交",
        user_id=user_id,
        icon="bi bi-chat-left-text",
        content=json.dumps(record, ensure_ascii=False, sort_keys=True),
    )
    db.session.add(log)
    db.session.commit()
    return record


def find_feedback(feedback_id: str) -> dict | None:
    """Find a receipt-safe record by its opaque public identifier."""

    if not FEEDBACK_ID_PATTERN.fullmatch(feedback_id or ""):
        return None

    marker = f'"feedback_id": "{feedback_id}"'
    logs = (
        SystemLog.query.filter(
            SystemLog.log_type == "反馈提交",
            SystemLog.content.like(f"%{marker}%"),
        )
        .order_by(SystemLog.id.desc())
        .all()
    )
    for log in logs:
        try:
            record = json.loads(log.content)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if record.get("feedback_id") == feedback_id:
            return record
    return None


def list_feedback(*, limit: int = 100) -> list[dict]:
    """Return recent structured records for the administrator review page."""

    safe_limit = max(1, min(int(limit), 200))
    logs = (
        SystemLog.query.filter_by(log_type="反馈提交")
        .order_by(SystemLog.id.desc())
        .limit(safe_limit)
        .all()
    )
    records = []
    for log in logs:
        try:
            record = json.loads(log.content)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or not record.get("feedback_id"):
            continue
        record["submitted_by"] = log.user_id
        records.append(record)
    return records
