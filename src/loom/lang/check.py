"""Type checker for Loom language."""
from __future__ import annotations

import math
from typing import Any, Optional

from . import ast
from .types import (
    FEATURE_TYPE, CIRCUIT_TYPE, MONITOR_TYPE,
    Type,
    SimpleType,
    CorpusType,
    TupleType,
    UnionType,
    NUMBER_TYPE,
    TARGET_TYPE,
    CORPUS_TYPE,
    CORPUS_CHAT_TYPE,
    CORPUS_PREF_TYPE,
    TOKENIZER_TYPE,
    MODEL_TYPE,
    EVALSET_TYPE,
    UNIT_TYPE,
    TypeError as LoomTypeError,
    types_match,
)


class TypeError(Exception):
    """Type checking error with position."""

    def __init__(self, msg: str, pos: Optional[ast.SourcePos] = None):
        self.msg = msg
        self.pos = pos
        if pos:
            super().__init__(f"Type error at {pos}: {msg}")
        else:
            super().__init__(f"Type error: {msg}")


class Warning:
    """Compiler warning."""

    def __init__(self, msg: str, pos: Optional[ast.SourcePos] = None):
        self.msg = msg
        self.pos = pos

    def __str__(self) -> str:
        if self.pos:
            return f"Warning at {self.pos}: {self.msg}"
        return f"Warning: {self.msg}"


class TypeChecker:
    """Type checker for Loom programs."""

    def __init__(self):
        self.env: dict[str, Type] = {}  # symbol table
        self.warnings: list[Warning] = []
        self.assertions: list[ast.Assert] = []  # for dead-result checking
        self.defined: set[str] = set()  # symbols that are defined
        self.used: set[str] = set()  # symbols that are used

    def check(self, program: ast.Program) -> dict[str, Type]:
        """Type check a program and return symbol table."""
        # First pass: collect definitions
        for stmt in program.stmts:
            if isinstance(stmt, ast.Decl):
                self.defined.add(stmt.name)
            elif isinstance(stmt, ast.Assign):
                if stmt.name not in self.defined:
                    raise TypeError(
                        f"Cannot assign to undefined symbol '{stmt.name}'", stmt.pos
                    )

        # Second pass: type check
        for stmt in program.stmts:
            self.check_stmt(stmt)

        # Check for dead results: models that are never asserted or exported
        self._check_dead_results(program)

        return self.env

    def check_stmt(self, stmt: ast.Stmt) -> None:
        """Type check a statement."""
        if isinstance(stmt, ast.Decl):
            self._check_decl(stmt)
        elif isinstance(stmt, ast.Assign):
            self._check_assign(stmt)
        elif isinstance(stmt, ast.Assert):
            self._check_assert(stmt)
        elif isinstance(stmt, ast.Export):
            self._check_export(stmt)
        elif isinstance(stmt, ast.Import):
            self._check_import(stmt)

    def _check_decl(self, decl: ast.Decl) -> None:
        """Check a declaration."""
        typ = self._infer_type(decl.expr)

        # Expected type from declaration
        expected_typ = self._type_from_keyword(decl.typ)

        # Check that inferred type matches expected
        if not types_match(expected_typ, typ):
            raise TypeError(
                f"Declaration '{decl.name}: {expected_typ}' has wrong type. "
                f"Expression has type {typ}",
                decl.pos,
            )

        self.env[decl.name] = typ

    def _check_assign(self, assign: ast.Assign) -> None:
        """Check an assignment."""
        if assign.name not in self.env:
            raise TypeError(
                f"Cannot assign to undefined symbol '{assign.name}'", assign.pos
            )

        typ = self._infer_type(assign.expr)
        expected_typ = self.env[assign.name]

        if not types_match(expected_typ, typ):
            raise TypeError(
                f"Assignment to '{assign.name}': expected {expected_typ}, got {typ}",
                assign.pos,
            )

        self.env[assign.name] = typ

    def _check_assert(self, assert_stmt: ast.Assert) -> None:
        """Check an assertion."""
        self.assertions.append(assert_stmt)

        left_type = self._infer_type(assert_stmt.left)
        right_type = self._infer_type(assert_stmt.right)

        # Right side should be a number
        if not isinstance(right_type, type(NUMBER_TYPE)):
            raise TypeError(
                f"Assertion RHS must be a number, got {right_type}",
                assert_stmt.pos,
            )

        # Check for impossible assertions
        self._check_assertion_possible(assert_stmt)

    def _check_export(self, export: ast.Export) -> None:
        """Check an export statement."""
        if export.name not in self.env:
            raise TypeError(f"Cannot export undefined symbol '{export.name}'", export.pos)

        # Export should be a model
        typ = self.env[export.name]
        if not isinstance(typ, SimpleType) or typ.name != "model":
            raise TypeError(
                f"Can only export models, but '{export.name}' has type {typ}",
                export.pos,
            )

        self.used.add(export.name)

    def _check_import(self, import_stmt: ast.Import) -> None:
        """Check an import statement."""
        # For now, just accept imports; actual loading happens in frontend
        pass

    def _infer_type(self, expr: ast.Expr) -> Type:
        """Infer the type of an expression."""
        if isinstance(expr, ast.Literal):
            return NUMBER_TYPE

        if isinstance(expr, ast.Ident):
            if expr.name in self.env:
                self.used.add(expr.name)
                return self.env[expr.name]
            # Allow built-in constants
            if expr.name in ["dpo", "ppo", "orpo", "slerp", "ties", "linear"]:
                return NUMBER_TYPE  # Placeholder for algorithm/config values
            raise TypeError(f"Undefined symbol '{expr.name}'", expr.pos)

        if isinstance(expr, ast.BinOp):
            return self._infer_binop_type(expr)

        if isinstance(expr, ast.Member):
            return self._infer_member_type(expr)

        if isinstance(expr, ast.Call):
            return self._infer_call_type(expr)

        raise TypeError(f"Cannot infer type of expression", expr.pos)

    def _infer_binop_type(self, binop: ast.BinOp) -> Type:
        """Infer type of binary operation."""
        left_type = self._infer_type(binop.left)
        right_type = self._infer_type(binop.right)

        if binop.op == "+":
            # corpus + corpus -> corpus
            if isinstance(left_type, CorpusType) and isinstance(right_type, CorpusType):
                return CORPUS_TYPE  # mixing corpora loses the kind
            raise TypeError(
                f"Cannot add {left_type} and {right_type}", binop.pos
            )

        if binop.op == "*":
            # corpus * number -> corpus
            if isinstance(left_type, CorpusType) and isinstance(right_type, type(NUMBER_TYPE)):
                return left_type  # preserve corpus kind
            # number * corpus -> corpus
            if isinstance(left_type, type(NUMBER_TYPE)) and isinstance(right_type, CorpusType):
                return right_type
            raise TypeError(
                f"Cannot multiply {left_type} and {right_type}", binop.pos
            )

        raise TypeError(f"Unknown binary operator: {binop.op}", binop.pos)

    def _infer_member_type(self, member: ast.Member) -> Type:
        """Infer type of member access."""
        obj_type = self._infer_type(member.object)

        # corpus.heldout -> evalset
        if isinstance(obj_type, CorpusType) and member.member == "heldout":
            return EVALSET_TYPE

        # corpus.split(frac) -> (corpus, corpus)
        if isinstance(obj_type, CorpusType) and member.member == "split":
            return TupleType([CORPUS_TYPE, CORPUS_TYPE])

        # corpus.filter(...) -> corpus
        if isinstance(obj_type, CorpusType) and member.member in ["filter", "dedup", "shuffle"]:
            return obj_type

        # Other members: assume method calls return void (error if called)
        raise TypeError(
            f"Unknown member '{member.member}' of {obj_type}", member.pos
        )

    def _infer_call_type(self, call: ast.Call) -> Type:
        """Infer type of function call."""
        path_str = ".".join(call.path)

        # Decode function signature and check args
        if path_str == "decoder":
            return TARGET_TYPE
        elif path_str == "load":
            return TARGET_TYPE

        elif path_str == "data.text":
            return CORPUS_TYPE
        elif path_str == "data.chat":
            return CORPUS_CHAT_TYPE
        elif path_str == "data.prefs":
            return CORPUS_PREF_TYPE

        # ---- mechanistic-interpretability primitives: the language's own verbs
        elif path_str == "concept":
            # concept(contrast=corpus) -> feature. A feature is defined by what
            # distinguishes it, so a contrast set is required.
            return FEATURE_TYPE
        elif path_str == "data.pairs":
            return CORPUS_TYPE
        elif path_str.startswith("std.features."):
            return FEATURE_TYPE
        elif path_str.startswith("std.circuits."):
            return CIRCUIT_TYPE
        elif path_str in ("construct", "learn_circuit"):
            return CIRCUIT_TYPE
        elif path_str == "read":
            # read(model, feature) -> monitor
            return MONITOR_TYPE
        elif path_str in ("amplify", "suppress"):
            # amplify(model, feature) -> model. A steering control without a bounded
            # side-effect is the failure mode the red-team found, so the block must
            # carry one.
            if call.block is not None:
                keys = set(call.block.items.keys())
                if keys and not any("side_effect" in k for k in keys):
                    raise TypeError(
                        f"{path_str}(...) has no side-effect bound. A control that is "
                        "not bounded can pass its effect gate by damaging the model; "
                        "add side_effect_max = <value> to the block.", call.pos)
            return MODEL_TYPE
        elif path_str == "install":
            # install(model, circuit) -> model
            return MODEL_TYPE
        elif path_str in ("probe_acc", "steer_effect", "circuit_gate"):
            return NUMBER_TYPE

        elif path_str == "tokenizer.bpe":
            return TOKENIZER_TYPE

        elif path_str == "pretrain":
            # pretrain(target, corpus, tokenizer) -> model
            self._check_call_args(call, ["target", "corpus", "tokenizer"], "pretrain")
            return MODEL_TYPE

        elif path_str == "finetune":
            # finetune(model, corpus[chat]) -> model
            self._check_call_args(call, ["model", "corpus[chat]"], "finetune")
            return MODEL_TYPE

        elif path_str == "align":
            # align(model, corpus[pref]) -> model
            self._check_call_args(call, ["model", "corpus[pref]"], "align")
            return MODEL_TYPE

        elif path_str == "graft":
            # graft(model, unit) -> model
            self._check_call_args(call, ["model", "unit"], "graft")
            return MODEL_TYPE

        elif path_str == "merge":
            # merge(model, model) -> model
            self._check_call_args(call, ["model", "model"], "merge")
            return MODEL_TYPE

        elif path_str == "quantize":
            # quantize(model) -> model
            self._check_call_args(call, ["model"], "quantize")
            return MODEL_TYPE

        elif path_str == "distill":
            # distill(model, model, corpus) -> model
            self._check_call_args(call, ["model", "model", "corpus"], "distill")
            return MODEL_TYPE

        elif path_str == "perplexity":
            # perplexity(model, corpus|evalset) -> number
            return NUMBER_TYPE

        elif path_str == "accuracy":
            # accuracy(model, evalset) -> number
            return NUMBER_TYPE

        elif path_str == "winrate":
            # winrate(model, baseline=...) -> number
            return NUMBER_TYPE

        elif path_str == "eval.blimp":
            # eval.blimp() -> evalset
            return EVALSET_TYPE

        elif path_str == "eval.jsonl":
            # eval.jsonl(path) -> evalset
            return EVALSET_TYPE

        elif path_str == "adamw":
            # adamw(lr=..., wd=...) -> optimizer config (just a dict)
            return NUMBER_TYPE  # Placeholder

        elif path_str == "cosine":
            # cosine(warmup=...) -> schedule config
            return NUMBER_TYPE  # Placeholder

        elif path_str == "dpo":
            # dpo -> alignment algorithm
            return NUMBER_TYPE  # Placeholder

        else:
            # Unknown function
            raise TypeError(f"Unknown function: {path_str}", call.pos)

    def _check_call_args(
        self, call: ast.Call, expected_types: list[str], func_name: str
    ) -> None:
        """Check that call arguments match expected types."""
        if len(call.args) != len(expected_types):
            raise TypeError(
                f"Function {func_name} expects {len(expected_types)} arguments, "
                f"got {len(call.args)}",
                call.pos,
            )

        for i, (arg, expected_str) in enumerate(zip(call.args, expected_types)):
            actual_type = self._infer_type(arg.value)
            expected_type = self._parse_type_str(expected_str)

            if not types_match(expected_type, actual_type):
                raise TypeError(
                    f"Argument {i} to {func_name}: expected {expected_type}, got {actual_type}",
                    arg.pos,
                )

    def _parse_type_str(self, s: str) -> Type:
        """Parse a type string like 'corpus[chat]' or 'model'."""
        if s == "corpus":
            return CORPUS_TYPE
        elif s == "corpus[chat]":
            return CORPUS_CHAT_TYPE
        elif s == "corpus[pref]":
            return CORPUS_PREF_TYPE
        elif s == "target":
            return TARGET_TYPE
        elif s == "tokenizer":
            return TOKENIZER_TYPE
        elif s == "model":
            return MODEL_TYPE
        elif s == "evalset":
            return EVALSET_TYPE
        elif s == "unit":
            return UNIT_TYPE
        elif s == "number":
            return NUMBER_TYPE
        else:
            raise ValueError(f"Unknown type string: {s}")

    def _type_from_keyword(self, keyword: str) -> Type:
        """Get the type from a type keyword."""
        if keyword == "target":
            return TARGET_TYPE
        elif keyword == "corpus":
            return CORPUS_TYPE
        elif keyword == "tokenizer":
            return TOKENIZER_TYPE
        elif keyword == "model":
            return MODEL_TYPE
        elif keyword == "evalset":
            return EVALSET_TYPE
        elif keyword == "unit":
            return UNIT_TYPE
        elif keyword == "feature":
            return FEATURE_TYPE
        elif keyword == "circuit":
            return CIRCUIT_TYPE
        elif keyword == "monitor":
            return MONITOR_TYPE
        else:
            raise ValueError(f"Unknown type keyword: {keyword}")

    def _check_assertion_possible(self, assert_stmt: ast.Assert) -> None:
        """Check if an assertion is possible (not contradictory)."""
        # Extract the metric name and threshold from a simple form like:
        # perplexity(m, held) < 30
        # This is a simple heuristic check

        if isinstance(assert_stmt.right, ast.Literal):
            threshold = assert_stmt.right.value
            if not isinstance(threshold, (int, float)):
                return

            # Check common metric ranges
            # perplexity >= 1, accuracy in [0,1], loss >= 0, etc.
            if isinstance(assert_stmt.left, ast.Call):
                func_name = ".".join(assert_stmt.left.path)
                if func_name == "perplexity":
                    if assert_stmt.op == "<" and threshold < 1:
                        raise TypeError(
                            f"Impossible assertion: perplexity < {threshold} (perplexity >= 1)",
                            assert_stmt.pos,
                        )
                elif func_name == "accuracy":
                    if assert_stmt.op == "<" and threshold <= 0:
                        raise TypeError(
                            f"Impossible assertion: accuracy < {threshold} (accuracy in [0,1])",
                            assert_stmt.pos,
                        )

    def _check_dead_results(self, program: ast.Program) -> None:
        """Check for dead results: models never exported or asserted."""
        models = set()
        for stmt in program.stmts:
            if isinstance(stmt, ast.Decl) and stmt.typ == "model":
                models.add(stmt.name)
            elif isinstance(stmt, ast.Assign):
                # Check if this is a model assignment (would need type info)
                pass

        # Mark used models
        for stmt in program.stmts:
            if isinstance(stmt, ast.Export):
                self.used.add(stmt.name)
            elif isinstance(stmt, ast.Assert):
                self._collect_used_idents(stmt.left)
                self._collect_used_idents(stmt.right)

        # Warn about unused models (not errors, just warnings)
        for model in models:
            if model not in self.used:
                self.warnings.append(
                    Warning(
                        f"Model '{model}' is defined but never exported or asserted",
                        None,
                    )
                )

    def _collect_used_idents(self, expr: ast.Expr) -> None:
        """Collect all identifiers used in an expression."""
        if isinstance(expr, ast.Ident):
            self.used.add(expr.name)
        elif isinstance(expr, ast.BinOp):
            self._collect_used_idents(expr.left)
            self._collect_used_idents(expr.right)
        elif isinstance(expr, ast.Member):
            self._collect_used_idents(expr.object)
        elif isinstance(expr, ast.Call):
            for arg in expr.args:
                self._collect_used_idents(arg.value)
