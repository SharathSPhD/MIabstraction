"""Intermediate representation: typed stage graph DAG."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Any, Optional, List

from . import ast
from .types import Type, SimpleType, CorpusType, NUMBER_TYPE


@dataclass
class IRNode:
    """A node in the stage graph DAG."""

    id: str  # unique identifier
    op: str  # operation: data.text, pretrain, etc.
    inputs: List[str] = field(default_factory=list)  # IDs of input nodes
    args: dict[str, Any] = field(default_factory=dict)  # operation arguments
    typ: Optional[str] = None  # output type
    cost: float = 0.0  # estimated compute cost in GPU-hours

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            "id": self.id,
            "op": self.op,
        }
        if self.inputs:
            d["inputs"] = self.inputs
        if self.args:
            d["args"] = self.args
        if self.typ:
            d["type"] = self.typ
        if self.cost > 0:
            d["cost"] = self.cost
        return d


@dataclass
class IRGraph:
    """Intermediate representation: a typed DAG of stages."""

    nodes: list[IRNode] = field(default_factory=list)
    targets: dict[str, Any] = field(default_factory=dict)  # backend, device_plan, etc.
    symbol_table: dict[str, str] = field(default_factory=dict)  # maps names to node IDs

    def add_node(self, node: IRNode) -> str:
        """Add a node and return its ID."""
        self.nodes.append(node)
        return node.id

    def to_json(self) -> str:
        """Serialize to JSON."""
        data = {
            "nodes": [n.to_dict() for n in self.nodes],
        }
        if self.targets:
            data["targets"] = self.targets
        return json.dumps(data, indent=2)

    def to_text(self) -> str:
        """Human-readable text representation."""
        lines = ["IR Stage Graph:"]
        lines.append("=" * 60)

        for node in self.nodes:
            line = f"  {node.id}: {node.op}"
            if node.inputs:
                line += f" <- {', '.join(node.inputs)}"
            if node.typ:
                line += f" : {node.typ}"
            if node.cost > 0:
                line += f" [cost: {node.cost:.2f} GPU-h]"
            lines.append(line)

            if node.args:
                for key, val in node.args.items():
                    lines.append(f"      {key} = {val}")

        lines.append("=" * 60)

        if self.targets:
            lines.append("Targets:")
            for key, val in self.targets.items():
                lines.append(f"  {key}: {val}")

        return "\n".join(lines)


class IRGenerator:
    """Generates IR from a type-checked AST."""

    def __init__(self):
        self.graph = IRGraph()
        self.node_counter = 0
        self.symbol_to_node: dict[str, str] = {}  # maps variable names to node IDs

    def generate(self, program: ast.Program) -> IRGraph:
        """Generate IR from an AST."""
        for stmt in program.stmts:
            self._lower_stmt(stmt)

        # Update symbol table in graph
        self.graph.symbol_table = self.symbol_to_node

        # Ensure topological order
        self._topological_sort()

        return self.graph

    def _topological_sort(self) -> None:
        """Sort nodes in topological order."""
        # Build adjacency info
        adj: dict[str, set[str]] = {n.id: set() for n in self.graph.nodes}
        in_degree: dict[str, int] = {n.id: 0 for n in self.graph.nodes}

        for node in self.graph.nodes:
            for inp_id in node.inputs:
                adj[inp_id].add(node.id)
                in_degree[node.id] += 1

        # Kahn's algorithm
        queue = [nid for nid, degree in in_degree.items() if degree == 0]
        sorted_ids = []

        while queue:
            nid = queue.pop(0)
            sorted_ids.append(nid)

            for neighbor in adj[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Reorder nodes
        id_to_node = {n.id: n for n in self.graph.nodes}
        self.graph.nodes = [id_to_node[nid] for nid in sorted_ids]

    def _genid(self) -> str:
        """Generate a unique node ID."""
        nid = f"n{self.node_counter}"
        self.node_counter += 1
        return nid

    def _lower_stmt(self, stmt: ast.Stmt) -> None:
        """Lower a statement to IR nodes."""
        if isinstance(stmt, ast.Decl):
            self._lower_decl(stmt)
        elif isinstance(stmt, ast.Assign):
            self._lower_assign(stmt)
        elif isinstance(stmt, ast.Export):
            self._lower_export(stmt)
        elif isinstance(stmt, ast.Assert):
            self._lower_assert(stmt)

    def _lower_decl(self, decl: ast.Decl) -> None:
        """Lower a declaration."""
        node_id = self._lower_expr(decl.expr)
        self.symbol_to_node[decl.name] = node_id

    def _lower_assign(self, assign: ast.Assign) -> None:
        """Lower an assignment."""
        node_id = self._lower_expr(assign.expr)
        self.symbol_to_node[assign.name] = node_id

    def _lower_export(self, export: ast.Export) -> None:
        """Lower an export statement."""
        if export.name in self.symbol_to_node:
            model_id = self.symbol_to_node[export.name]
            nid = self._genid()
            node = IRNode(
                id=nid,
                op="export",
                inputs=[model_id],
                args={"path": export.path},
                typ="export",
            )
            self.graph.add_node(node)

    def _lower_assert(self, assert_stmt: ast.Assert) -> None:
        """Lower an assertion."""
        left_id = self._lower_expr(assert_stmt.left)
        right_id = self._lower_expr(assert_stmt.right)

        nid = self._genid()
        node = IRNode(
            id=nid,
            op="assert",
            inputs=[left_id, right_id],
            args={"op": assert_stmt.op},
            typ="assert",
        )
        self.graph.add_node(node)

    def _lower_expr(self, expr: ast.Expr) -> str:
        """Lower an expression, return its node ID."""
        if isinstance(expr, ast.Literal):
            return self._lower_literal(expr)
        elif isinstance(expr, ast.Ident):
            return self._lower_ident(expr)
        elif isinstance(expr, ast.BinOp):
            return self._lower_binop(expr)
        elif isinstance(expr, ast.Member):
            return self._lower_member(expr)
        elif isinstance(expr, ast.Call):
            return self._lower_call(expr)
        else:
            raise ValueError(f"Unknown expression type: {type(expr)}")

    def _lower_literal(self, lit: ast.Literal) -> str:
        """Lower a literal to a constant node."""
        nid = self._genid()
        node = IRNode(
            id=nid,
            op="literal",
            args={"value": lit.value},
            typ="number",
        )
        self.graph.add_node(node)
        return nid

    def _lower_ident(self, ident: ast.Ident) -> str:
        """Lower an identifier (must be defined)."""
        if ident.name in self.symbol_to_node:
            return self.symbol_to_node[ident.name]
        # Handle built-in constants
        if ident.name in ["dpo", "ppo", "orpo", "slerp", "ties", "linear", "adamw", "cosine"]:
            nid = self._genid()
            node = IRNode(
                id=nid,
                op=f"constant.{ident.name}",
                args={"value": ident.name},
                typ="config",
            )
            self.graph.add_node(node)
            return nid
        raise ValueError(f"Undefined identifier: {ident.name}")

    def _lower_binop(self, binop: ast.BinOp) -> str:
        """Lower a binary operation."""
        left_id = self._lower_expr(binop.left)
        right_id = self._lower_expr(binop.right)

        nid = self._genid()
        node = IRNode(
            id=nid,
            op=f"binop.{binop.op}",
            inputs=[left_id, right_id],
            typ="corpus" if binop.op in ["+", "*"] else "number",
        )
        self.graph.add_node(node)
        return nid

    def _lower_member(self, member: ast.Member) -> str:
        """Lower member access (e.g., corpus.heldout)."""
        obj_id = self._lower_expr(member.object)

        nid = self._genid()
        op = f"member.{member.member}"
        if member.args:
            # Method call with arguments
            arg_ids = [self._lower_expr(arg.value) for arg in member.args]
            node = IRNode(
                id=nid,
                op=op,
                inputs=[obj_id] + arg_ids,
                typ="evalset" if member.member == "heldout" else "corpus",
            )
        else:
            # Property access
            node = IRNode(
                id=nid,
                op=op,
                inputs=[obj_id],
                typ="evalset" if member.member == "heldout" else "corpus",
            )

        self.graph.add_node(node)
        return nid

    def _lower_call(self, call: ast.Call) -> str:
        """Lower a function call."""
        func_name = ".".join(call.path)

        # Collect input node IDs
        input_ids = []
        args = {}

        for arg in call.args:
            arg_id = self._lower_expr(arg.value)
            if arg.name:
                # Named argument
                args[arg.name] = arg_id
            else:
                # Positional argument
                input_ids.append(arg_id)

        # Add block arguments if present
        if call.block:
            for key, val in call.block.items.items():
                if isinstance(val, ast.Literal):
                    args[key] = val.value
                else:
                    # More complex block values would need more lowering
                    args[key] = self._lower_expr(val)

        # Infer output type
        output_type = self._infer_call_output_type(func_name)

        nid = self._genid()
        node = IRNode(
            id=nid,
            op=func_name,
            inputs=input_ids,
            args=args,
            typ=output_type,
        )

        self.graph.add_node(node)
        return nid

    def _infer_call_output_type(self, func_name: str) -> str:
        """Infer output type of a function."""
        if func_name in [
            "decoder",
            "load",
        ]:
            return "target"
        elif func_name in ["data.text", "data.chat", "data.prefs"]:
            return "corpus"
        elif func_name == "tokenizer.bpe":
            return "tokenizer"
        elif func_name in [
            "pretrain",
            "finetune",
            "align",
            "graft",
            "merge",
            "quantize",
            "distill",
        ]:
            return "model"
        elif func_name in ["perplexity", "accuracy"]:
            return "number"
        elif func_name in ["eval.blimp", "eval.jsonl"]:
            return "evalset"
        else:
            return "unknown"
