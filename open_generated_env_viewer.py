from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path


def main() -> None:
    libdir = sysconfig.get_config_var("LIBDIR")
    env = os.environ.copy()
    if libdir:
        existing = env.get("DYLD_FALLBACK_LIBRARY_PATH")
        paths = [libdir]
        if existing:
            paths.append(existing)
        paths.extend(["/usr/local/lib", "/usr/lib"])
        env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(paths)

    script = Path(__file__).with_name("open_generated_env.py")
    command = ["mjpython", str(script), *sys.argv[1:]]
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
