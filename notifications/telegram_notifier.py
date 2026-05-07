from __future__ import annotations

import os
from typing import Any

import requests

from core.logging_utils import get_logger

logger = get_logger(__name__)


def telegram_credentials(config: Any) -> tuple[str, str]:
    telegram_config = getattr(config, "telegram", None)
    token = (
        os.getenv("TELEGRAM_BOT_TOKEN")
        or os.getenv("TELEGRAM_TOKEN")
        or getattr(telegram_config, "bot_token", "")
        or getattr(config, "TELEGRAM_BOT_TOKEN", "")
        or getattr(config, "TELEGRAM_TOKEN", "")
    )
    chat_id = (
        os.getenv("TELEGRAM_CHAT_ID")
        or getattr(telegram_config, "chat_id", "")
        or getattr(config, "TELEGRAM_CHAT_ID", "")
    )
    return token, chat_id


def send_telegram_msg(config: Any, text: str) -> None:
    token, chat_id = telegram_credentials(config)
    if not token or not chat_id:
        return
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=5,
        )
        response.raise_for_status()
    except Exception as exc:
        logger.error("Telegram Error: %s", exc)


def test_telegram_connection(config: Any) -> dict[str, Any]:
    token, chat_id = telegram_credentials(config)
    if not token or not chat_id:
        return {"ok": False, "reason": "Missing TELEGRAM_BOT_TOKEN/TELEGRAM_TOKEN or TELEGRAM_CHAT_ID."}
    try:
        bot_response = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
        bot_payload = bot_response.json()
        if not bot_payload.get("ok"):
            return {"ok": False, "reason": f"Bot token rejected: {bot_payload}"}
        message_response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": "AQRS V3 Telegram intelligence feed is connected."},
            timeout=10,
        )
        message_payload = message_response.json()
        if not message_payload.get("ok"):
            return {"ok": False, "reason": f"Chat/message rejected: {message_payload}", "bot": bot_payload.get("result", {})}
        return {
            "ok": True,
            "bot": bot_payload.get("result", {}),
            "chat_id": chat_id,
            "message_id": message_payload.get("result", {}).get("message_id"),
        }
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def send_trade_alert(config: Any, alert_type: str, payload: dict[str, Any]) -> None:
    _, chat_id = telegram_credentials(config)
    if not chat_id:
        return

    def _clean(value: Any) -> str:
        return str(value).replace("_", " ")

    text = (
        f"*AQRS V3 {alert_type}*\n\n"
        f"*Symbol:* {_clean(payload.get('symbol', 'XAUUSD'))}\n"
        f"*System:* {_clean(payload.get('system', payload.get('signal', 'UNKNOWN')))}\n"
        f"*Side:* {_clean(payload.get('confirmed_signal', payload.get('side', 'UNKNOWN'))).upper()}\n"
        f"*Price:* {float(payload.get('entry_price', payload.get('price', 0.0))):.2f}\n"
        f"*SL / TP:* {float(payload.get('stop_loss', 0.0)):.2f} / {float(payload.get('take_profit', 0.0)):.2f}\n"
        f"*Lifecycle:* {_clean(payload.get('lifecycle_state', 'UNKNOWN'))}\n"
        f"*HTF Bias:* {_clean(payload.get('htf_bias', 'NEUTRAL'))}\n"
        f"*Liquidity:* {_clean(payload.get('liquidity_event', 'NONE'))}\n"
        f"*Continuation:* {float(payload.get('continuation_strength', 0.0)):.1f}\n"
        f"*Exhaustion:* {float(payload.get('exhaustion_score', 0.0)):.1f}\n"
        f"*Trap Prob:* {float(payload.get('trap_probability', 0.0)):.1f}\n"
        f"*Alpha / Flow:* {float(payload.get('alpha_score', 0.0)):.1f} / {float(payload.get('flow_score', 0.0)):.1f}\n"
        f"*MTF Align:* {float(payload.get('multi_tf_alignment_score', 0.0)):.1f}\n"
        f"*Exit State:* {_clean(payload.get('exit_state', 'NONE'))}"
    )
    send_telegram_msg(config, text)
