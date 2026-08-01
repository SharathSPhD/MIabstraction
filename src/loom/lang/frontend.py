"""Frontend: compile_source and compile_weave entry points."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from .lexer import Lexer
from .parser import Parser
from .check import TypeChecker, TypeError as CheckTypeError
from .ir import IRGenerator, IRGraph
from . import ast


class CompileError(Exception):
    """Compilation error with context."""

    pass


def compile_source(path: str | Path) -> IRGraph:
    """
    Compile a .loom source file to IR.

    Returns:
        IRGraph: The intermediate representation (typed DAG).

    Raises:
        CompileError: If compilation fails (syntax, type, or semantic errors).
    """
    path = Path(path)
    if not path.exists():
        raise CompileError(f"File not found: {path}")

    source = path.read_text()

    try:
        # Lexical analysis
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # Parsing
        parser = Parser(tokens)
        program = parser.parse()

        # Type checking
        checker = TypeChecker()
        checker.check(program)

        # Emit warnings
        for warning in checker.warnings:
            print(f"  {warning}")

        # IR generation
        generator = IRGenerator()
        ir = generator.generate(program)

        return ir

    except SyntaxError as e:
        raise CompileError(f"Parse error in {path}: {e}")
    except CheckTypeError as e:
        raise CompileError(f"Type error in {path}: {e}")


def compile_weave(path: str | Path) -> IRGraph:
    """
    Compile a weave.yaml file to IR (desugaring declarative to imperative).

    The OLD weave.yaml format is converted into a Loom program that:
    - Loads or creates a target
    - Applies finetune/align stages per skills
    - Asserts gates

    Returns:
        IRGraph: The intermediate representation.

    Raises:
        CompileError: If compilation fails.
    """
    path = Path(path)
    if not path.exists():
        raise CompileError(f"File not found: {path}")

    try:
        weave_data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise CompileError(f"YAML error in {path}: {e}")

    # Desugar weave.yaml to Loom program
    program_text = _desugar_weave(weave_data)

    # Compile the desugared program
    try:
        lexer = Lexer(program_text)
        tokens = lexer.tokenize()

        parser = Parser(tokens)
        program = parser.parse()

        checker = TypeChecker()
        checker.check(program)

        generator = IRGenerator()
        ir = generator.generate(program)

        return ir

    except SyntaxError as e:
        raise CompileError(f"Parse error in desugared weave: {e}")
    except CheckTypeError as e:
        raise CompileError(f"Type error in desugared weave: {e}")


def _desugar_weave(weave_data: dict) -> str:
    """
    Convert weave.yaml to Loom program text.

    Example:
        skills: [{name: chat, kind: instruct}]
        gates:  {chat: {winrate: ">0.55"}}

    Desugars to:
        target arch = load("default");
        ...finetune...
        assert winrate(m, baseline) > 0.55;
        export m to "result";
    """
    lines = []

    # Target: default to loading a base model
    target_spec = weave_data.get("target", {})
    if isinstance(target_spec, dict):
        # Load a specific model
        model_name = target_spec.get("model", "Qwen/Qwen2.5-0.5B")
        lines.append(f'target arch = load("{model_name}");')
    else:
        lines.append(f'target arch = load("Qwen/Qwen2.5-0.5B");')

    # Foundation (optional)
    foundation = weave_data.get("foundation")
    if foundation:
        lines.append(f'corpus train_data = data.text("{foundation.get("corpus", "babylm:strict")}");')
        lines.append('model m = pretrain(arch, train_data, tk) {')
        if "budget_hours" in foundation:
            lines.append(f'  budget = {foundation["budget_hours"]};')
        lines.append("};")
    else:
        # No foundation, start with loaded model
        lines.append("model m = arch;")

    # Skills → finetune/align stages
    skills = weave_data.get("skills", [])
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        name = skill.get("name", "unknown")
        kind = skill.get("kind", "instruct")

        if kind in ["instruct", "sft"]:
            # Finetune stage
            corpus_path = f'data.chat("{name}_data.jsonl")'
            lines.append(f"model m = finetune(m, {corpus_path}) {{")
            lines.append(f'  skill = "{name}";')
            lines.append("};")
        elif kind in ["preference", "dpo", "rlhf"]:
            # Align stage
            corpus_path = f'data.prefs("{name}_prefs.jsonl")'
            lines.append(f"model m = align(m, {corpus_path}) {{")
            lines.append(f'  algo = dpo;')
            lines.append("};")

    # Gates → assertions
    gates = weave_data.get("gates", {})
    for gate_name, gate_spec in gates.items():
        if not isinstance(gate_spec, dict):
            continue
        for metric, threshold_str in gate_spec.items():
            # Parse threshold like ">0.55" or "<0.1"
            if isinstance(threshold_str, str):
                op = threshold_str[0]
                threshold = threshold_str[1:]
            else:
                op = ">"
                threshold = threshold_str

            lines.append(f"assert {metric}(m) {op} {threshold};")

    # Export
    lines.append('export m to "result";')

    return "\n".join(lines)
