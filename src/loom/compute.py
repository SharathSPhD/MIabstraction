"""Compute-target planner: decide where to run a Loom job.

Rules:
- Pretraining-from-scratch and throughput-bound => rtx5090 (fast, 32GB)
- Jobs needing >32GB or frozen models => local_gb10 (slow, 128GB unified)

Dispatches to RTX 5090 via bash scripts in the rtx5090-connect skill:
  - submit_job.sh <local_job_dir> <remote_job_name> [entrypoint.py]
  - status.sh <remote_job_name>
  - fetch_results.sh <remote_job_name> <local_dest_dir> [remote_subpath]
  - cleanup.sh <remote_job_name>
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ComputePlan:
    target: str  # "local_gb10" | "rtx5090"
    rationale: str


def plan_target(job_config: dict) -> ComputePlan:
    """Decide where to run a Loom job.

    Args:
        job_config: Dict with keys like:
            - is_pretraining (bool): from-scratch pretraining run?
            - n_params (int): model size in parameters
            - max_memory_gb (int): estimated peak memory
            - throughput_critical (bool): is speed important?

    Returns:
        ComputePlan with target and plain-English rationale.
    """
    is_pretraining = job_config.get("is_pretraining", False)
    n_params = job_config.get("n_params", 10_000_000)
    max_memory_gb = job_config.get("max_memory_gb", 8)
    throughput_critical = job_config.get("throughput_critical", True)

    # Decision logic
    if max_memory_gb > 32:
        return ComputePlan(
            target="local_gb10",
            rationale=(
                f"Job needs ~{max_memory_gb}GB peak memory, which exceeds RTX 5090's 32GB. "
                "Using local GB10 (128GB unified, slower but enough headroom)."
            ),
        )

    if is_pretraining and throughput_critical:
        return ComputePlan(
            target="rtx5090",
            rationale=(
                f"Foundation pretraining is throughput-bound ({n_params//1e6:.1f}M params). "
                "RTX 5090 provides 32GB and fast compute; budget constraints fit within "
                f"{max_memory_gb}GB. Using RTX 5090."
            ),
        )

    # Default: local is safe if it fits in memory
    if max_memory_gb <= 32:
        return ComputePlan(
            target="rtx5090",
            rationale=(
                f"Job uses {max_memory_gb}GB and is throughput-sensitive. "
                "RTX 5090 (32GB, fast) is sufficient."
            ),
        )

    return ComputePlan(
        target="local_gb10",
        rationale="Conservative: using local GB10 for safety.",
    )


def get_rtx5090_skill_path() -> Path:
    """Path to the RTX 5090 skill scripts."""
    skill_root = Path.home() / ".claude" / "skills" / "rtx5090-connect"
    if not skill_root.exists():
        raise FileNotFoundError(
            f"RTX 5090 skill not found at {skill_root}. "
            "Install it via: skills add rtx5090-connect"
        )
    return skill_root


def submit_to_rtx5090(
    job_dir: Path,
    job_name: str,
    entrypoint: str = "train.py",
    skill_path: Path | None = None,
) -> str:
    """Submit a job to the RTX 5090.

    Args:
        job_dir: Local directory containing job files
        job_name: Remote job name (becomes directory on RTX 5090)
        entrypoint: Entry script (default "train.py")
        skill_path: Path to rtx5090-connect skill (auto-detected if None)

    Returns:
        First lines of train.log (confirmation the job started)

    Raises:
        RuntimeError: if submission fails
    """
    if skill_path is None:
        skill_path = get_rtx5090_skill_path()

    script = skill_path / "scripts" / "submit_job.sh"
    try:
        result = subprocess.run(
            ["bash", str(script), str(job_dir), job_name, entrypoint],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"submit_job.sh failed:\n{result.stderr}\n{result.stdout}"
            )
        return result.stdout
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"submit_job.sh timed out after 30s")


def get_status(job_name: str, skill_path: Path | None = None) -> str:
    """Check the status of a remote job.

    Args:
        job_name: Remote job name
        skill_path: Path to rtx5090-connect skill

    Returns:
        Status output (tailed logs + GPU stats)
    """
    if skill_path is None:
        skill_path = get_rtx5090_skill_path()

    script = skill_path / "scripts" / "status.sh"
    try:
        result = subprocess.run(
            ["bash", str(script), job_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] status.sh timed out after 10s for job {job_name}"


def fetch_results(
    job_name: str,
    local_dest: Path,
    remote_subpath: str = "checkpoints",
    skill_path: Path | None = None,
) -> None:
    """Fetch results from a remote job.

    Args:
        job_name: Remote job name
        local_dest: Local directory to copy results to
        remote_subpath: Remote subpath to fetch (default "checkpoints")
        skill_path: Path to rtx5090-connect skill
    """
    if skill_path is None:
        skill_path = get_rtx5090_skill_path()

    script = skill_path / "scripts" / "fetch_results.sh"
    try:
        result = subprocess.run(
            ["bash", str(script), job_name, str(local_dest), remote_subpath],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"fetch_results.sh failed:\n{result.stderr}\n{result.stdout}"
            )
        print(result.stdout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"fetch_results.sh timed out after 120s")


def cleanup_job(job_name: str, skill_path: Path | None = None, confirm: bool = True) -> None:
    """Delete a remote job directory.

    Args:
        job_name: Remote job name
        skill_path: Path to rtx5090-connect skill
        confirm: Ask for confirmation interactively if True
    """
    if skill_path is None:
        skill_path = get_rtx5090_skill_path()

    script = skill_path / "scripts" / "cleanup.sh"
    env = {}
    if not confirm:
        env["CONFIRM"] = "yes"

    try:
        result = subprocess.run(
            ["bash", str(script), job_name],
            capture_output=True,
            text=True,
            timeout=30,
            env={**subprocess.os.environ, **env},
        )
        if result.returncode != 0:
            raise RuntimeError(f"cleanup.sh failed:\n{result.stderr}")
        print(result.stdout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"cleanup.sh timed out after 30s")
