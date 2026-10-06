"""Repository paths and reproducible package snapshots, independent of cwd."""
from pathlib import Path
import shutil
import sys


def project_root():
    """Find the editable checkout (or the root of a frozen source snapshot)."""
    for root in Path(__file__).resolve().parents:
        if (root / "pyproject.toml").is_file() and (root / "uv.lock").is_file():
            return root
    raise FileNotFoundError("Project files not found. Use the editable install via uv sync.")


def dependency_lock():
    """Prefer the lock belonging to the active uv environment."""
    lock = Path(sys.prefix).parent / "uv.lock"
    return lock if lock.is_file() else project_root() / "uv.lock"


def snapshot_source(destination):
    """Copy the package as an importable package, preserving relative imports."""
    destination = Path(destination) / "lunarlander_rl"
    destination.mkdir(parents=True, exist_ok=True)
    for source in Path(__file__).resolve().parent.glob("*.py"):
        target = destination / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    return destination


def frozen_command(source, module):
    """Run with cwd=source so the frozen package overrides the editable install.

    Legacy, flat snapshots remain executable with their original script layout.
    """
    source = Path(source).resolve()
    if (source / "lunarlander_rl" / "__init__.py").is_file():
        return [sys.executable, "-m", f"lunarlander_rl.{module}"]
    return [sys.executable, str(source / f"{module}.py")]
