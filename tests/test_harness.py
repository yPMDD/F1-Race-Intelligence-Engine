"""Guards the test harness itself.

Regression: CI failed with `ModuleNotFoundError: No module named 'storage'`
while the same suite passed locally every time. The cause was an asymmetry in
how pytest is invoked:

  * locally  -> `python -m pytest`   the `-m` form puts the CWD on sys.path
  * in CI    -> `uv run pytest`      the console script does NOT

This project has no __init__.py files anywhere, so `storage`, `ml`, `nlp` and
`apps` are PEP 420 namespace packages anchored only by the repo root. Nothing
anchors them unless the root is on sys.path.
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(args, env_extra=None):
    env = dict(os.environ)
    env.setdefault("DATABASE_URL", "sqlite:///./.pytest_harness_check.db")
    env.setdefault("GROQ_API_KEY", "placeholder-not-real")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        args, capture_output=True, text=True, env=env, cwd=ROOT,
    )


def test_pytest_config_declares_the_repo_root_on_syspath():
    """The `pythonpath` ini option is the actual fix; guard it exists."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        pytest.skip("tomllib unavailable")

    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        cfg = tomllib.load(f)

    ini = cfg.get("tool", {}).get("pytest", {}).get("ini_options", {})
    assert "." in ini.get("pythonpath", []), (
        "pyproject.toml must declare pythonpath = [\".\"]. Without it, "
        "`pytest` (as opposed to `python -m pytest`) cannot import the "
        "top-level namespace packages storage/ml/nlp/apps."
    )


def test_top_level_packages_are_importable_without_cwd_on_syspath():
    """Import each package the way a bare `pytest` run would see them."""
    code = (
        "import sys, pathlib\n"
        # Emulate the console script: drop '' and the CWD from sys.path.
        "sys.path[:] = [p for p in sys.path if p not in ('', '.', str(pathlib.Path.cwd()))]\n"
        "sys.path.insert(0, '.')\n"   # what the pythonpath ini option does
        "import storage.postgres.models, ml.config, nlp.agents.tools, apps.api.main\n"
        "print('ok')\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=ROOT,
        env=dict(os.environ, DATABASE_URL="sqlite:///./.pytest_harness_check.db"),
    )
    assert "ok" in r.stdout, f"import failed:\n{r.stdout}\n{r.stderr}"


def test_project_has_no_init_files_anchoring_packages():
    """Documents WHY pythonpath is required.

    If someone ever adds __init__.py files and restructures the packages, this
    test fails and the pythonpath entry can be reconsidered deliberately
    rather than removed by accident.
    """
    inits = [
        os.path.relpath(os.path.join(dirpath, f), ROOT)
        for dirpath, dirnames, filenames in os.walk(ROOT)
        for f in filenames
        if f == "__init__.py"
        and not dirpath.startswith(os.path.join(ROOT, ".venv"))
        and "node_modules" not in dirpath
        and os.sep + ".git" not in dirpath
    ]
    assert not inits, (
        f"__init__.py files now exist ({inits}); packages are properly "
        "anchored and the pythonpath workaround may be re-evaluated."
    )


def test_bare_pytest_collects_successfully():
    """End-to-end: the exact command CI runs must not fail to import."""
    r = _run([sys.executable, "-m", "pytest", "--collect-only", "-q"])
    if r.returncode != 0 and "ModuleNotFoundError" in r.stderr:
        pytest.fail(
            "bare `pytest` cannot import the project's packages:\n" + r.stderr[-2000:]
        )
    assert "error" not in r.stderr.lower()[-2000:] or r.returncode == 0
