"""Start-up material shared by provider plug-ins whose hosts start no SSH daemon themselves."""

#: Started first by the start-up script of a provider that runs the image as it is, with no SSH
#: daemon of its own (RunPod's pods; the container on a Verda VM): the pool reaches its hosts over
#: SSH. It makes sure one runs — installed from the image's own package
#: manager when missing, key login only (the pool's key is put in place by the script itself) —
#: in the background, so a slow install never holds up the engine. Unverified on a live pod.
SSH_DAEMON = (
    "( if ! command -v sshd >/dev/null 2>&1 && [ ! -x /usr/sbin/sshd ]; then "
    "{ apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server; } "
    "|| apk add --no-cache openssh-server; fi; "
    # Host keys made here, never the image's: a public image's baked-in keys are known to anyone
    # who pulls it, which would defeat the pool's trust in the first key it sees.
    # Made once: a container started again (a restart policy, a reboot) keeps the keys the pool
    # already trusts, where new ones would leave it refusing the host for good.
    "mkdir -p /run/sshd && { [ -f /etc/ssh/.gpm-host-keys ] || { rm -f /etc/ssh/ssh_host_*_key "
    "/etc/ssh/ssh_host_*_key.pub && ssh-keygen -A && touch /etc/ssh/.gpm-host-keys; }; } && "
    "/usr/sbin/sshd -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no "
    "-o PermitRootLogin=prohibit-password ) >/var/log/gpm-sshd.log 2>&1 &"
)
