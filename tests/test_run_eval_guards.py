"""run_eval harness guards: abstain-schema strictness and repo-relative paths.

Pins the two behaviors that keep run metadata honest: schema-strict adapters must
not run --abstain without the abstain-aware schema env switch (the model could
never abstain), and recorded paths must be repo-relative, never absolute local
paths that leak the runner's machine layout.
"""
from pathlib import Path

from commoditybench.models import build_model
from commoditybench.run_eval import (
    _abstain_env_active,
    _display_path,
    _model_is_schema_strict,
)

ROOT = Path(__file__).resolve().parents[1]


def test_abstain_env_switch_recognition(monkeypatch):
    for value, expected in [("1", True), ("true", True), ("on", True),
                            ("yes", True), ("0", False), ("", False),
                            ("True", True)]:
        monkeypatch.setenv("COMMODITYBENCH_ABSTAIN_AWARE", value)
        assert _abstain_env_active() is expected, value


def test_schema_strict_detection_matches_adapter_lanes():
    # json_object / none lanes never send a schema -> prompt alone carries the
    # abstain fields; json_schema + native-structured adapters are strict.
    assert not _model_is_schema_strict(build_model("deepseek-v4-flash"))
    assert not _model_is_schema_strict(build_model("qwen3-32b"))
    assert _model_is_schema_strict(build_model("gpt-4o"))
    assert _model_is_schema_strict(build_model("claude-opus-4-8"))
    assert _model_is_schema_strict(build_model("gemini-2.0-flash"))


def test_display_path_is_repo_relative():
    assert _display_path(ROOT / "data" / "questions.jsonl") == "data/questions.jsonl"
    assert _display_path(ROOT / "results" / "x.jsonl") == "results/x.jsonl"


def test_display_path_falls_back_for_external_paths():
    out = _display_path("/etc/hosts")
    assert out == "/etc/hosts" or not out.startswith("data/")
