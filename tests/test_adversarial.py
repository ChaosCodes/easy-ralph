"""
Tests for Evaluator adversarial testing feature.

Validates prompt construction, new data fields, and file I/O
for adversarial findings / responses.
"""

import os
from pathlib import Path

import pytest

from ralph_sdk.evaluator import (
    EvaluationResult,
    Metric,
    MetricType,
    build_evaluator_prompt,
    read_adversarial_response,
)
from ralph_sdk.pool import AUDITS_DIR, RALPH_DIR, init_ralph_dir
from ralph_sdk.prompts import (
    EVALUATOR_ADVERSARIAL_SECTION,
    WORKER_ADVERSARIAL_INVESTIGATION_SECTION,
)


# ---------------------------------------------------------------------------
# Prompt constants exist
# ---------------------------------------------------------------------------


def test_evaluator_adversarial_section_exists():
    """EVALUATOR_ADVERSARIAL_SECTION is a non-empty string with key placeholders."""
    assert isinstance(EVALUATOR_ADVERSARIAL_SECTION, str)
    assert len(EVALUATOR_ADVERSARIAL_SECTION) > 100
    assert "{audits_dir}" in EVALUATOR_ADVERSARIAL_SECTION
    assert "{task_id}" in EVALUATOR_ADVERSARIAL_SECTION
    assert "{attempt}" in EVALUATOR_ADVERSARIAL_SECTION


def test_worker_adversarial_investigation_section_exists():
    """WORKER_ADVERSARIAL_INVESTIGATION_SECTION is a non-empty string with key placeholders."""
    assert isinstance(WORKER_ADVERSARIAL_INVESTIGATION_SECTION, str)
    assert len(WORKER_ADVERSARIAL_INVESTIGATION_SECTION) > 100
    assert "{audits_dir}" in WORKER_ADVERSARIAL_INVESTIGATION_SECTION
    assert "{task_id}" in WORKER_ADVERSARIAL_INVESTIGATION_SECTION
    assert "{findings_content}" in WORKER_ADVERSARIAL_INVESTIGATION_SECTION


# ---------------------------------------------------------------------------
# EvaluationResult new fields
# ---------------------------------------------------------------------------


def test_evaluation_result_has_adversarial_fields():
    """EvaluationResult has adversarial_findings and adversarial_findings_path."""
    result = EvaluationResult(
        task_id="T001",
        overall_passed=True,
        overall_score=90,
    )
    assert result.adversarial_findings == ""
    assert result.adversarial_findings_path == ""


def test_evaluation_result_adversarial_fields_settable():
    """adversarial fields can be set on EvaluationResult."""
    result = EvaluationResult(
        task_id="T001",
        overall_passed=True,
        overall_score=90,
        adversarial_findings="some findings",
        adversarial_findings_path="/tmp/test/findings.md",
    )
    assert result.adversarial_findings == "some findings"
    assert result.adversarial_findings_path == "/tmp/test/findings.md"


# ---------------------------------------------------------------------------
# build_evaluator_prompt includes adversarial section
# ---------------------------------------------------------------------------


def test_evaluator_prompt_includes_adversarial_section():
    """When audits_dir is provided, prompt includes adversarial instructions."""
    metrics = [Metric("test_metric", MetricType.HARD, "A test metric")]
    prompt = build_evaluator_prompt(
        task_id="T001",
        goal="Test goal",
        task_detail="Test detail",
        metrics=metrics,
        audits_dir="/tmp/audits",
        attempt_number=1,
    )
    assert "Adversarial Verification Phase" in prompt
    assert "/tmp/audits" in prompt
    assert "adversarial_T001_1" in prompt


def test_evaluator_prompt_without_adversarial():
    """When audits_dir is empty, prompt does NOT include adversarial section."""
    metrics = [Metric("test_metric", MetricType.HARD, "A test metric")]
    prompt = build_evaluator_prompt(
        task_id="T001",
        goal="Test goal",
        task_detail="Test detail",
        metrics=metrics,
    )
    assert "Adversarial Verification Phase" not in prompt


def test_evaluator_prompt_includes_previous_response():
    """When previous_adversarial_responses is provided, prompt includes it."""
    metrics = [Metric("test_metric", MetricType.HARD, "A test metric")]
    prompt = build_evaluator_prompt(
        task_id="T001",
        goal="Test goal",
        task_detail="Test detail",
        metrics=metrics,
        audits_dir="/tmp/audits",
        previous_adversarial_responses="Worker rebutted finding 1 as false positive.",
    )
    assert "Worker's Previous Adversarial Response" in prompt
    assert "Worker rebutted finding 1 as false positive." in prompt


def test_evaluator_prompt_no_previous_response():
    """When previous_adversarial_responses is None, no response section appears."""
    metrics = [Metric("test_metric", MetricType.HARD, "A test metric")]
    prompt = build_evaluator_prompt(
        task_id="T001",
        goal="Test goal",
        task_detail="Test detail",
        metrics=metrics,
        audits_dir="/tmp/audits",
    )
    assert "Worker's Previous Adversarial Response" not in prompt


def test_evaluator_prompt_no_previous_scores():
    """Anti-anchoring: prompt must NOT contain specific previous score numbers."""
    metrics = [Metric("test_metric", MetricType.HARD, "A test metric")]
    prompt = build_evaluator_prompt(
        task_id="T001",
        goal="Test goal",
        task_detail="Test detail",
        metrics=metrics,
        attempt_number=4,
        previous_scores=[87.0, 90.0, 93.0],
    )
    # Should include attempt history section
    assert "Attempt History" in prompt
    assert "#4" in prompt
    # Must NOT include specific score numbers (anti-anchoring)
    assert "87" not in prompt
    assert "90" not in prompt
    assert "93" not in prompt
    assert "Previous scores" not in prompt
    assert "Score trend" not in prompt
    # Should include anti-anchoring instructions
    assert "Evaluate the code AS-IS" in prompt
    assert "current quality" in prompt


# ---------------------------------------------------------------------------
# read_adversarial_response
# ---------------------------------------------------------------------------


def test_read_adversarial_response_exists(tmp_path):
    """read_adversarial_response returns file content when file exists."""
    init_ralph_dir(str(tmp_path))
    audits = tmp_path / AUDITS_DIR
    audits.mkdir(parents=True, exist_ok=True)

    response_file = audits / "response_T001_2.md"
    response_file.write_text("## Findings Response\n\nREBUTTED: not a real bug")

    result = read_adversarial_response("T001", 2, str(tmp_path))
    assert result is not None
    assert "REBUTTED" in result


def test_read_adversarial_response_missing(tmp_path):
    """read_adversarial_response returns None when file doesn't exist."""
    init_ralph_dir(str(tmp_path))

    result = read_adversarial_response("T001", 2, str(tmp_path))
    assert result is None


# ---------------------------------------------------------------------------
# AUDITS_DIR constant and init
# ---------------------------------------------------------------------------


def test_audits_dir_constant():
    """AUDITS_DIR is .ralph/audits."""
    assert AUDITS_DIR == ".ralph/audits"


def test_init_ralph_dir_creates_audits(tmp_path):
    """init_ralph_dir creates the audits directory."""
    init_ralph_dir(str(tmp_path))
    assert (tmp_path / AUDITS_DIR).is_dir()


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def test_adversarial_section_formats_correctly():
    """EVALUATOR_ADVERSARIAL_SECTION formats with all placeholders."""
    formatted = EVALUATOR_ADVERSARIAL_SECTION.format(
        audits_dir="/tmp/audits",
        task_id="T002",
        attempt=3,
    )
    assert "/tmp/audits/adversarial_T002_3.py" in formatted
    assert "/tmp/audits/adversarial_T002_3.md" in formatted


def test_worker_investigation_section_formats_correctly():
    """WORKER_ADVERSARIAL_INVESTIGATION_SECTION formats with all placeholders."""
    formatted = WORKER_ADVERSARIAL_INVESTIGATION_SECTION.format(
        audits_dir="/tmp/audits",
        task_id="T002",
        attempt=3,
        findings_content="Finding 1: bug in lookup table",
    )
    assert "/tmp/audits/response_T002_3.md" in formatted
    assert "Finding 1: bug in lookup table" in formatted
