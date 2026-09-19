"""The dead-man timer that every rented host carries.

docs/spec/supervisor.md §7. If the supervisor crashes, or its machine sleeps or loses its
network, lease limits stop being enforced while the instance keeps billing. So the timer lives
**on the host**, independent of anything off-host, and ends the instance itself.

Two conditions, both required: **no supervisor heartbeat** and **no inference**. Both, because
the router is a separate process — a host still serving requests while the supervisor restarts
is doing useful work and must be left alone.

The credential is the provider's **instance-scoped** one, which can only act on that instance.
**The account credential is never placed on a rented machine** (threat model T5).
"""

from __future__ import annotations

from typing import Optional

#: Where the timer keeps its two timestamps on the host.
STATE_DIR = "/var/run/gpm"

_SCRIPT = """#!/bin/sh
# GPM dead-man timer. Ends this instance if the pool goes silent.
# It carries no account credential: {fire_note}
set -u
STATE_DIR="${{GPM_STATE_DIR:-{state_dir}}}"
WINDOW="${{GPM_DEADMAN_SECONDS:-{window_s}}}"
POLL="${{GPM_DEADMAN_POLL_SECONDS:-{poll_s}}}"
PORT="${{GPM_ENGINE_PORT:-{engine_port}}}"

mkdir -p "$STATE_DIR"
HEARTBEAT="$STATE_DIR/heartbeat"
ACTIVITY="$STATE_DIR/activity"
touch "$HEARTBEAT" "$ACTIVITY"

# Portable mtime: GNU stat, then BSD stat.
mtime() {{ stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo 0; }}

while true; do
    sleep "$POLL"

    # Any established connection to the engine counts as inference in progress. Sampling can
    # miss a request that begins and ends between polls, which is why the window is minutes
    # and the poll is seconds.
    # /proc/net/tcp is always there on Linux; ss and netstat often are not in a slim image
    # (seen live). State 01 is ESTABLISHED; the local port is hex in column 2.
    if [ -r /proc/net/tcp ]; then
        HEXPORT=$(printf '%04X' "$PORT")
        if cat /proc/net/tcp /proc/net/tcp6 2>/dev/null | awk -v p=":$HEXPORT" 'NR>1 && $4=="01" && index($2,p)==length($2)-length(p)+1 {{found=1}} END {{exit !found}}'; then
            touch "$ACTIVITY"
        fi
    elif command -v ss >/dev/null 2>&1; then
        ss -tn state established 2>/dev/null | grep -q ":$PORT" && touch "$ACTIVITY"
    elif command -v netstat >/dev/null 2>&1; then
        netstat -tn 2>/dev/null | grep ESTABLISHED | grep -q ":$PORT" && touch "$ACTIVITY"
    fi

    NOW=$(date +%s)
    SINCE_HEARTBEAT=$((NOW - $(mtime "$HEARTBEAT")))
    SINCE_ACTIVITY=$((NOW - $(mtime "$ACTIVITY")))

    if [ "$SINCE_HEARTBEAT" -ge "$WINDOW" ] && [ "$SINCE_ACTIVITY" -ge "$WINDOW" ]; then
        echo "gpm: no supervisor for ${{SINCE_HEARTBEAT}}s and no inference for \
${{SINCE_ACTIVITY}}s; ending this instance" >&2
        {fire_command}
        exit 0
    fi
done
"""


def build_script(
    fire_command: str,
    *,
    window_s: int,
    poll_s: int = 30,
    engine_port: int = 11434,
    state_dir: str = STATE_DIR,
) -> str:
    """The on-host loop. `fire_command` comes from the provider — only it knows how one of its
    instances ends itself, and with which instance-scoped credential."""
    return _SCRIPT.format(
        fire_command=fire_command,
        fire_note="it uses the provider's instance-scoped credential.",
        window_s=int(window_s),
        poll_s=int(poll_s),
        engine_port=int(engine_port),
        state_dir=state_dir,
    )


def heartbeat_command(state_dir: str = STATE_DIR) -> str:
    """What the supervisor runs on the host each pass, over the connection it already holds."""
    return f"mkdir -p {state_dir} && touch {state_dir}/heartbeat"


def install_public_key_command(public_key: str, ssh_user: str = "root") -> str:
    """Let the pool's own key open the host, whatever keys the account has registered.

    A public key is not a secret, so it may travel in the start-up script; the private half
    never leaves the supervisor's machine.
    """
    home = "/root" if ssh_user == "root" else f"/home/{ssh_user}"
    key = public_key.strip().replace("'", "")
    return (
        f"mkdir -p {home}/.ssh && chmod 700 {home}/.ssh && "
        f"grep -qxF '{key}' {home}/.ssh/authorized_keys 2>/dev/null || "
        f"echo '{key}' >> {home}/.ssh/authorized_keys; chmod 600 {home}/.ssh/authorized_keys"
    )


def onstart_script(
    fire_command: str,
    *,
    window_s: int,
    engine_port: int = 11434,
    poll_s: int = 30,
    state_dir: str = STATE_DIR,
    public_key: Optional[str] = None,
    ssh_user: str = "root",
    extra: Optional[str] = None,
) -> str:
    """The instance's start-up script: arm the timer first, then everything else.

    Armed first on purpose — a host that fails later in its start-up must still be able to end
    itself rather than bill until somebody notices. The pool's public key goes in next, so the
    supervisor can reach the host to heartbeat the timer and forward the engine.
    """
    script = build_script(
        fire_command, window_s=window_s, poll_s=poll_s, engine_port=engine_port, state_dir=state_dir
    )
    lines = [
        "mkdir -p " + state_dir,
        f"cat > {state_dir}/deadman.sh <<'GPM_DEADMAN_EOF'",
        script,
        "GPM_DEADMAN_EOF",
        f"chmod +x {state_dir}/deadman.sh",
        f"nohup {state_dir}/deadman.sh >{state_dir}/deadman.log 2>&1 &",
    ]
    if public_key:
        lines.append(install_public_key_command(public_key, ssh_user))
    if extra:
        lines.append(extra)
    return "\n".join(lines)
