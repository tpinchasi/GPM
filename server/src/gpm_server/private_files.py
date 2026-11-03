"""Whether a file that holds a secret is this user's alone — the key files, the client CA's key,
the provider credentials (D163).

On macOS and Linux its permission bits say so, and each caller reads them. On Windows they mean
nothing to Python, which reports every ordinary file as readable by everyone, and the standard
library reads no access control list; so, as the host agent does (D154, host-agent.md §2.3), the
place decides: a file under the user's profile is the user's, the system's and administrators'
alone by the rules Windows gives that folder, and one elsewhere — a folder made at the root of a
drive, say — usually lets every signed-in user read or change it.

The rule is the agent's, copied rather than imported: the server and the agent are separate
packages, and neither depends on the other.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

#: Read where it is used, so that a test can play Windows on another system.
WINDOWS = os.name == "nt"

#: Where Windows says this user's profile is: the profile itself, and where its local
#: application data was moved, if it was.
PROFILE_VARIABLES = ("USERPROFILE", "LOCALAPPDATA")


def outside_profile(path: str, environ: Mapping[str, str] = os.environ, paths: Any = os.path) -> Optional[str]:
    """Why a secret at `path` is not to be trusted on Windows, or None: it lies under this user's
    profile. Links are followed first, so a junction out of the profile counts as where it leads.
    `paths` is the path module the rules are read with."""
    roots = []
    for name in PROFILE_VARIABLES:
        value = environ.get(name) or ""
        if value and paths.isabs(value):
            roots.append(paths.normcase(paths.realpath(value)))
    if not roots:
        return (
            "cannot be checked: on Windows the pool reads its keys and credentials only from under "
            "your profile, and USERPROFILE does not say where that is"
        )
    where = paths.normcase(paths.realpath(path))
    for root in roots:
        try:
            if paths.commonpath([root, where]) == root:
                return None
        except ValueError:  # another drive
            continue
    profile = environ.get("USERPROFILE") or environ.get("LOCALAPPDATA")
    return (
        f"is outside your profile ({profile}). On Windows the pool reads its keys and credentials "
        "only from under your profile: a folder elsewhere — one made at the root of a drive, say — "
        f"usually lets every user of this machine read or change what is in it. Move it under {profile}"
    )
