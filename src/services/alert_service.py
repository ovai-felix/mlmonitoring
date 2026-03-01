"""Alert notification service with Slack and Email routing by severity."""
import json
import logging
import smtplib
from email.mime.text import MIMEText
from urllib.request import Request, urlopen

from src.config import settings

logger = logging.getLogger(__name__)


class AlertService:
    """Routes alerts to Slack and/or Email based on severity.

    Severity routing:
        critical -> Slack + Email
        warning  -> Slack only
        info     -> log only
    """

    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "info",
        context: dict | None = None,
    ) -> dict:
        """Send an alert routed by severity. Never raises."""
        results = {"severity": severity, "channels": []}

        if severity == "critical":
            if self._send_slack(title, message, severity, context):
                results["channels"].append("slack")
            if self._send_email(title, message, severity, context):
                results["channels"].append("email")
        elif severity == "warning":
            if self._send_slack(title, message, severity, context):
                results["channels"].append("slack")
        # info -> log only
        logger.info("[ALERT:%s] %s — %s", severity.upper(), title, message)
        return results

    def _send_slack(
        self,
        title: str,
        message: str,
        severity: str,
        context: dict | None,
    ) -> bool:
        """POST to Slack webhook. Returns True on success."""
        url = settings.slack_webhook_url
        if not url:
            return False

        icon = {"critical": ":rotating_light:", "warning": ":warning:"}.get(severity, ":information_source:")
        payload = {
            "text": f"{icon} *[{severity.upper()}] {title}*\n{message}",
        }
        if context:
            fields = "\n".join(f"• {k}: {v}" for k, v in context.items())
            payload["text"] += f"\n{fields}"

        try:
            data = json.dumps(payload).encode()
            req = Request(url, data=data, headers={"Content-Type": "application/json"})
            urlopen(req, timeout=10)
            return True
        except Exception:
            logger.exception("Failed to send Slack alert: %s", title)
            return False

    def _send_email(
        self,
        title: str,
        message: str,
        severity: str,
        context: dict | None,
    ) -> bool:
        """Send email via SMTP. Returns True on success."""
        if not settings.smtp_host or not settings.alert_email_to:
            return False

        body = f"[{severity.upper()}] {title}\n\n{message}"
        if context:
            body += "\n\nContext:\n" + "\n".join(f"  {k}: {v}" for k, v in context.items())

        msg = MIMEText(body)
        msg["Subject"] = f"[ML Monitor {severity.upper()}] {title}"
        msg["From"] = settings.smtp_user or "mlmonitor@localhost"
        msg["To"] = settings.alert_email_to

        try:
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as server:
                if settings.smtp_user and settings.smtp_pass:
                    server.starttls()
                    server.login(settings.smtp_user, settings.smtp_pass)
                server.send_message(msg)
            return True
        except Exception:
            logger.exception("Failed to send email alert: %s", title)
            return False


alert_service = AlertService()
