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

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # a type only, so the timer keeps no import of the provider package
    from .providers.base import SelfTerminateRequest

#: Where the timer keeps its two timestamps on the host.
STATE_DIR = "/var/run/gpm"
#: The script that starts the engine, and starts it again when its settings change (D56).
ENGINE_RESTART = f"{STATE_DIR}/restart-engine.sh"
ENGINE_ENV = f"{STATE_DIR}/engine.env"

_SCRIPT = """#!/bin/sh
# GPM dead-man timer. Ends this instance if the pool goes silent.
# It carries no account credential: {fire_note}
# The call it makes is the provider's; how it makes it depends on what this machine has.
set -u
STATE_DIR="${{GPM_STATE_DIR:-{state_dir}}}"
WINDOW="${{GPM_DEADMAN_SECONDS:-{window_s}}}"
POLL="${{GPM_DEADMAN_POLL_SECONDS:-{poll_s}}}"
PORT="${{GPM_ENGINE_PORT:-{engine_port}}}"

mkdir -p "$STATE_DIR"

# The call this instance makes to end itself, as data rather than as a command line.
cat > "$STATE_DIR/terminate.json" <<'GPM_REQUEST_EOF'
{request_json}
GPM_REQUEST_EOF

# How this machine can make an HTTPS call. Resolved *now*, while the pool is watching, rather
# than in twenty minutes when nothing is left to tell (D71): the first engine image carries
# neither curl nor wget, and a timer that cannot call the provider is no timer at all.
if command -v curl >/dev/null 2>&1; then HTTP_CLIENT=curl
elif command -v wget >/dev/null 2>&1; then HTTP_CLIENT=wget
elif command -v python3 >/dev/null 2>&1; then HTTP_CLIENT=python3
else
    # Last resort, and only at arm time: one package, quietly, best effort.
    (apt-get update -qq && apt-get install -y -qq curl) >/dev/null 2>&1 || true
    if command -v curl >/dev/null 2>&1; then HTTP_CLIENT=curl; else HTTP_CLIENT=none; fi
fi
echo "$HTTP_CLIENT" > "$STATE_DIR/http_client"
[ "$HTTP_CLIENT" = none ] && echo "gpm: no http client on this machine; the timer will stop \
the container instead of ending the instance" >&2

fire() {{
    case "$HTTP_CLIENT" in
    curl)    {curl_call} ;;
    wget)    {wget_call} ;;
    python3) {python_call} ;;
    *)
        # It cannot reach the provider, so it does the one thing it can: stop the container.
        # Billing for the accelerator ends, the instance shows as stopped, and a pool that is
        # alive destroys it on its next pass.
        kill -TERM 1 2>/dev/null || halt -f 2>/dev/null || true ;;
    esac
}}

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
        fire
        exit 0
    fi
done
"""


def _shell_quote(value: str) -> str:
    """Single-quoted for `sh`, with any quote of its own closed and reopened."""
    return "'" + value.replace("'", "'\\''") + "'"


def _calls(request: "SelfTerminateRequest", state_dir: str) -> dict[str, str]:
    """One provider request, in each client the host might have.

    The request itself travels as **data**: written to a file at arm time and read back when
    the timer fires. Nothing is interpolated into a command line, so a URL or a header value
    can hold anything without becoming shell. Header values may name an environment variable
    the provider injected, which each form expands on the host.
    """
    import json

    headers = dict(request.headers)
    curl = ["curl -sS -X " + request.method]
    wget = ["wget -q -O - --method=" + request.method]
    for key, value in headers.items():
        curl.append(f'-H "{key}: {value}"')
        wget.append(f'--header="{key}: {value}"')
    if request.body:
        curl.append("-d " + _shell_quote(request.body))
        wget.append("--body-data=" + _shell_quote(request.body))
    curl.append(f'"{request.url}"')
    wget.append(f'"{request.url}"')

    program = (
        "import json, os, urllib.request\n"
        f'd = json.load(open("{state_dir}/terminate.json"))\n'
        "h = {k: os.path.expandvars(v) for k, v in d[\"headers\"].items()}\n"
        "b = d.get(\"body\")\n"
        "r = urllib.request.Request(os.path.expandvars(d[\"url\"]), method=d[\"method\"],\n"
        "                           headers=h, data=b.encode() if b else None)\n"
        "try:\n"
        "    urllib.request.urlopen(r, timeout=30).read()\n"
        "except Exception as exc:\n"
        "    print(\"gpm: self-terminate failed:\", exc)\n"
    )
    return {
        "curl_call": " ".join(curl),
        "wget_call": " ".join(wget),
        "python_call": "python3 -c " + _shell_quote(program),
        "request_json": json.dumps(
            {
                "method": request.method,
                "url": request.url,
                "headers": headers,
                "body": request.body,
            }
        ),
    }


def build_script(
    request: "SelfTerminateRequest",
    *,
    window_s: int,
    poll_s: int = 30,
    engine_port: int = 11434,
    state_dir: str = STATE_DIR,
) -> str:
    """The on-host loop. The request comes from the provider — only it knows how one of its
    instances ends itself, and with which instance-scoped credential (D71)."""
    return _SCRIPT.format(
        **_calls(request, state_dir),
        fire_note="it uses the provider's instance-scoped credential.",
        window_s=int(window_s),
        poll_s=int(poll_s),
        engine_port=int(engine_port),
        state_dir=state_dir,
    )


def engine_restart_script(engine_start: str, *, state_dir: str = STATE_DIR) -> str:
    """Stop the engine this script last started, then start it again with today's settings.

    The settings arrive as an environment file the agent writes (D41: bounded whole numbers,
    turned into the engine's own variable names by the agent, never by the pool). The engine's
    own start-up line is the operator's, unchanged, so nothing here knows what an engine is.
    """
    return f"""#!/bin/sh
# GPM: start the engine, or start it again with what is in {state_dir}/engine.env.
set -u
PIDFILE="{state_dir}/engine.pid"

if [ -r "$PIDFILE" ]; then
    OLD=$(cat "$PIDFILE" 2>/dev/null || echo)
    if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
        kill -TERM "$OLD" 2>/dev/null || true
        # Give it a moment to put its models down before anything else is started.
        i=0
        while kill -0 "$OLD" 2>/dev/null && [ "$i" -lt 30 ]; do sleep 1; i=$((i + 1)); done
        kill -KILL "$OLD" 2>/dev/null || true
    fi
fi

if [ -r "{state_dir}/engine.env" ]; then
    set -a
    . "{state_dir}/engine.env"
    set +a
fi

{engine_start}
echo $! > "$PIDFILE"
"""


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
    request: "SelfTerminateRequest",
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
        request, window_s=window_s, poll_s=poll_s, engine_port=engine_port, state_dir=state_dir
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
        # The engine is started through a script rather than inline, so the same command can
        # start it again later with different settings — which is what resizing a running host
        # comes down to (D56). The operator's `engine_start` is reused verbatim; the pool never
        # invents a way to run an engine.
        lines.extend(
            [
                f"cat > {ENGINE_RESTART} <<'GPM_ENGINE_EOF'",
                engine_restart_script(extra, state_dir=state_dir),
                "GPM_ENGINE_EOF",
                f"chmod +x {ENGINE_RESTART}",
                f"sh {ENGINE_RESTART}",
            ]
        )
    return "\n".join(lines)
