from __future__ import annotations

import os
import smtplib
import sys
from email.message import EmailMessage
from typing import Optional

ENV_PREFIX = "OOM_MONITOR_"


class EmailNotifier:
    def __init__(
        self,
        *,
        to_address: str,
        from_address: Optional[str] = None,
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_starttls: Optional[bool] = None,
    ) -> None:
        self.to_address = to_address
        self.from_address = from_address or username or to_address
        self.smtp_server = smtp_server or os.getenv(f"{ENV_PREFIX}SMTP_SERVER")
        self.smtp_port = smtp_port or _safe_int(os.getenv(f"{ENV_PREFIX}SMTP_PORT"), default=587)
        self.username = username or os.getenv(f"{ENV_PREFIX}SMTP_USER")
        self.password = password or os.getenv(f"{ENV_PREFIX}SMTP_PASSWORD")
        if use_starttls is None:
            env_value = os.getenv(f"{ENV_PREFIX}SMTP_STARTTLS", "true").lower()
            use_starttls = env_value not in {"0", "false", "no"}
        self.use_starttls = use_starttls

    @property
    def enabled(self) -> bool:
        return bool(self.smtp_server and self.to_address)

    def send(self, subject: str, body: str) -> None:
        if not self.enabled:
            print("EmailNotifier disabled; missing SMTP configuration", file=sys.stderr)
            return
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.from_address
        message["To"] = self.to_address
        message.set_content(body)

        with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30) as server:
            if self.use_starttls:
                server.starttls()
            if self.username and self.password:
                server.login(self.username, self.password)
            server.send_message(message)


def _safe_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def create_notifier(to_address: Optional[str]) -> Optional[EmailNotifier]:
    if not to_address:
        return None
    return EmailNotifier(to_address=to_address)
