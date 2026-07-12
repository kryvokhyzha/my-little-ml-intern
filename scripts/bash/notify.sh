#!/usr/bin/env bash
# Milestone notifier — Telegram + Slack, formatted per event.
# Usage: notify.sh <event> "<message>" ["<experiment>"]
# Events: plan_ready | code_ready | train_started | train_done (alias: train_finished)
#         | error | blocker | approval_required
# Unknown events still send (generic 🔔 card) but warn on stderr — prefer the list above.
# Message may be multi-line: "- " bullet lines are rendered as 🔹 diamond bullets
# in both channels (write plain "- ", the script prettifies).
# Experiment (3rd arg, optional) falls back to $NOTIFY_EXPERIMENT.
# Project name from $PROJECT_NAME / $NOTIFY_PROJECT, else the repo directory name.
# Channels from env (or project-root .env): TG_BOT_TOKEN+TG_CHAT_ID,
# SLACK_WEBHOOK_URL or SLACK_BOT_TOKEN+SLACK_CHANNEL_ID.
# Missing env => silent per-channel no-op. NOTIFY_DRY_RUN=1 prints the composed
# messages instead of sending. ALWAYS exits 0 so callers never crash on a notify failure.

set -u

event="${1:-event}"
message="${2:-}"
experiment="${3:-${NOTIFY_EXPERIMENT:-}}"

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

# Prettify plain "- " bullet lines into diamond bullets on the cards.
message="$(printf '%s' "$message" | sed 's/^- /🔹 /')"

project="${PROJECT_NAME:-${NOTIFY_PROJECT:-$(basename "$project_root")}}"
host="$(hostname -s 2>/dev/null || echo unknown)"
# Time only, pinned to UTC — the messenger already shows the full local timestamp,
# and cards from remote VMs would otherwise mix timezones in one thread.
stamp="$(date -u '+%H:%M UTC' 2>/dev/null || echo '')"

case "$event" in
  plan_ready)
    emoji="📋"
    label="Plan ready"
    next="Review the plan, then launch (budget gate first)."
    ;;
  code_ready)
    emoji="🛠️"
    label="Code ready"
    next="Smoke-test before any long run."
    ;;
  train_started)
    emoji="🚀"
    label="Training started"
    next="Monitor metrics & alerts; verify when done."
    ;;
  train_done | train_finished)
    emoji="✅"
    label="Training complete"
    next="Run the verify gate before trusting results."
    ;;
  error)
    emoji="❌"
    label="Error"
    next="Read the logs, write a postmortem, decide on a retry."
    ;;
  blocker)
    emoji="🚧"
    label="Blocked — needs you"
    next="A human decision is required to continue."
    ;;
  approval_required)
    emoji="⏸️"
    label="Approval required"
    next="Proceeding on the assumed default — reply if wrong."
    ;;
  *)
    emoji="🔔"
    # Title-case the raw event name (loss_spike -> Loss Spike) so the card stays readable.
    label="$(printf '%s' "$event" | tr '_' ' ' | awk '{for (i = 1; i <= NF; i++) $i = toupper(substr($i, 1, 1)) substr($i, 2)} 1')"
    next=""
    echo "notify.sh: unknown event '$event' — sending a generic card (see the usage header for known events)" >&2
    ;;
esac

html_escape() {
  local s="$1"
  s="${s//&/&amp;}"
  s="${s//</&lt;}"
  s="${s//>/&gt;}"
  printf '%s' "$s"
}

json_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/\\r}"
  s="${s//$'\t'/\\t}"
  printf '%s' "$s"
}

# Telegram (parse_mode=HTML): bold header, code experiment, italic next/meta.
tg="${emoji} <b>$(html_escape "$label")</b>"$'\n'"<b>Project:</b> $(html_escape "$project")"
[ -n "$experiment" ] && tg+=" · <b>Exp:</b> <code>$(html_escape "$experiment")</code>"
[ -n "$message" ] && tg+=$'\n\n'"$(html_escape "$message")"
[ -n "$next" ] && tg+=$'\n\n'"➡️ <i>$(html_escape "$next")</i>"
tg+=$'\n\n'"<i>$(html_escape "${host} · ${stamp}")</i>"

# Slack (mrkdwn): & < > escaped, then the whole payload JSON-escaped.
sk="${emoji} *$(html_escape "$label")*"$'\n'"*Project:* $(html_escape "$project")"
[ -n "$experiment" ] && sk+=" · *Exp:* \`$(html_escape "$experiment")\`"
[ -n "$message" ] && sk+=$'\n\n'"$(html_escape "$message")"
[ -n "$next" ] && sk+=$'\n\n'"➡️ _$(html_escape "$next")_"
sk+=$'\n\n'"_$(html_escape "${host} · ${stamp}")_"

if [ -n "${NOTIFY_DRY_RUN:-}" ]; then
  printf '%s\n%s\n%s\n%s\n' "--- telegram ---" "$tg" "--- slack ---" "$sk"
  exit 0
fi

# Token-bearing values (bot-token URLs, Authorization headers) go through curl's stdin
# config (`--config -`) so they never show up on the process list.

# --- Telegram ---
if [ -n "${TG_BOT_TOKEN:-}" ] && [ -n "${TG_CHAT_ID:-}" ]; then
  curl -fsS -m 10 --config - \
    -d "chat_id=${TG_CHAT_ID}" \
    -d "parse_mode=HTML" \
    -d "disable_web_page_preview=true" \
    --data-urlencode "text=${tg}" \
    > /dev/null 2>&1 <<EOF || echo "notify.sh: telegram send failed" >&2
url = "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage"
EOF
fi

# --- Slack ---
escaped_text="$(json_escape "$sk")"
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
