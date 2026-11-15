"""Environment check: is a usable Java (17+) installed, can we write a temp folder?"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MIN_JAVA = 17


@dataclass(frozen=True)
class JavaInfo:
    executable: str
    major: int
    version_text: str
    from_path: bool  # True when plain "java" on PATH is the one we found


def parse_java_major(text: str) -> int | None:
    """``openjdk version "21.0.10"`` -> 21, ``"1.8.0_392"`` -> 8."""
    match = re.search(r'version "(\d+)(?:\.(\d+))?', text)
    if not match:
        return None
    major = int(match.group(1))
    return int(match.group(2)) if major == 1 and match.group(2) else major


def _probe(executable: str) -> tuple[int, str] | None:
    try:
        proc = subprocess.run([executable, "-version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    text = proc.stderr or proc.stdout
    major = parse_java_major(text)
    if major is None:
        return None
    match = re.search(r'version "([^"]+)"', text)
    return major, match.group(1) if match else str(major)


def _candidates() -> list[tuple[str, bool]]:
    found: list[tuple[str, bool]] = []
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        found.append((str(Path(java_home) / "bin" / "java"), False))
    on_path = shutil.which("java")
    if on_path:
        found.append((os.path.realpath(on_path), True))
    for name in ("openjdk@21", "openjdk@17"):  # Homebrew JDKs are often not registered with macOS
        found.append((f"/opt/homebrew/opt/{name}/bin/java", False))
        found.append((f"/usr/local/opt/{name}/bin/java", False))
    try:
        proc = subprocess.run(["/usr/libexec/java_home"], capture_output=True, text=True, timeout=10)
        if proc.returncode == 0 and proc.stdout.strip():
            found.append((str(Path(proc.stdout.strip()) / "bin" / "java"), False))
    except (OSError, subprocess.SubprocessError):
        pass
    return found


def find_java() -> JavaInfo | None:
    """First Java >= 17 among: JAVA_HOME, PATH, Homebrew paths, /usr/libexec/java_home."""
    seen: set[str] = set()
    for executable, from_path in _candidates():
        if executable in seen or not os.path.exists(executable):
            continue
        seen.add(executable)
        probed = _probe(executable)
        if probed and probed[0] >= MIN_JAVA:
            return JavaInfo(executable, probed[0], probed[1], from_path)
    return None


def apply_java_home(info: JavaInfo) -> None:
    """Point JAVA_HOME at the Java we found so Spark launches the same one."""
    real = Path(os.path.realpath(info.executable))
    if info.from_path and not os.environ.get("JAVA_HOME"):
        return  # Spark will find "java" on PATH by itself
    os.environ["JAVA_HOME"] = str(real.parent.parent)


def tmp_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


def python_version() -> str:
    return ".".join(str(p) for p in sys.version_info[:3])


def pyspark_version() -> str:
    import pyspark

    return pyspark.__version__
