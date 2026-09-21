"""The GPM host agent.

docs/spec/host-agent.md. Runs on a host an operator has delegated to a pool. The pool dials
it; it never dials the pool, is never on the request path, and its protocol is a closed list
of verbs — there is no "run this".
"""

#: The protocol version this agent speaks, as the first path segment (`/agent/v1/...`).
PROTOCOL_VERSION = "1"
__version__ = "0.2.2"
