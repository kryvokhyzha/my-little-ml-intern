#!/usr/bin/env bash
# Milestone notifier — Telegram + Slack.
# Usage: notify.sh <event> "<message>"
# Events: plan_ready | code_ready | train_started | train_done | error | blocker | approval_required
# Channels from env (or project-root .env): TG_BOT_TOKEN+TG_CHAT_ID,
# SLACK_WEBHOOK_URL or SLACK_BOT_TOKEN+SLACK_CHANNEL_ID.
# Missing env => silent per-channel no-op. ALWAYS exits 0 so callers never crash on notification failure.

set -u

event="${1:-event}"
message="${2:-}"

case "$event" in
  plan_ready | code_ready | train_started | train_done | error | blocker | approval_required) ;;
  *) echo "notify.sh: unknown event '${event}' — sending anyway" >&2 ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
env_file="${project_root}/.env"
if [ -f "$env_file" ]; then
  # A dotenv line referencing an unset var must not kill the script under `set -u`.
  set +u
  # shellcheck disable=SC1090
  set -a
  . "$env_file"
  set +a
  set -u
fi

host="$(hostname -s 2>/dev/null || echo unknown)"
text="[ml-intern@${host}] ${event}: ${message}"

json_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/\\r}"
  s="${s//$'\t'/\\t}"
  printf '%s' "$s"
}

# Token-bearing values (bot-token URLs, Authorization headers) go through curl's stdin
# config (`--config -`) so they never show up on the process list.

# --- Telegram ---
if [ -n "${TG_BOT_TOKEN:-}" ] && [ -n "${TG_CHAT_ID:-}" ]; then
  curl -fsS -m 10 --config - \
    -d "chat_id=${TG_CHAT_ID}" \
    --data-urlencode "text=${text}" \
    > /dev/null 2>&1 <<EOF || echo "notify.sh: telegram send failed" >&2
url = "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage"
EOF
fi

# --- Slack ---
escaped_text="$(json_escape "$text")"
if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
  curl -fsS -m 10 --config - \
    -H "Content-type: application/json" \
    -d "{\"text\":\"${escaped_text}\"}" \
    > /dev/null 2>&1 <<EOF || echo "notify.sh: slack webhook send failed" >&2
url = "${SLACK_WEBHOOK_URL}"
EOF
elif [ -n "${SLACK_BOT_TOKEN:-}" ] && [ -n "${SLACK_CHANNEL_ID:-}" ]; then
  escaped_channel="$(json_escape "$SLACK_CHANNEL_ID")"
  curl -fsS -m 10 --config - \
    -H "Content-type: application/json; charset=utf-8" \
    -d "{\"channel\":\"${escaped_channel}\",\"text\":\"${escaped_text}\"}" \
    > /dev/null 2>&1 <<EOF || echo "notify.sh: slack send failed" >&2
url = "https://slack.com/api/chat.postMessage"
header = "Authorization: Bearer ${SLACK_BOT_TOKEN}"
EOF
fi

exit 0
