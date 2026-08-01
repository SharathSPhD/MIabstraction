"""AST node definitions for Loom language."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any

from .lexer import SourcePos


@dataclass
class ASTNode:
    """Base class for all AST nodes with position tracking."""
    pos: SourcePos


# Expressions

@dataclass
class Literal(ASTNode):
    """Literal value: number, string."""
    value: Any  # float or str


@dataclass
class Ident(ASTNode):
    """Identifier."""
    name: str


@dataclass
class BinOp(ASTNode):
    """Binary operation: left op right."""
    op: str  # '+' or '*'
    left: Expr
    right: Expr


@dataclass
class Member(ASTNode):
    """Member access: object.member or object.method(...)."""
    object: Expr
    member: str
    args: Optional[list[Expr]] = None  # if method call


@dataclass
class Call(ASTNode):
    """Function call: path(args) with optional block."""
    path: list[str]  # e.g., ["data", "text"] for data.text(...)
    args: list[Arg] = field(default_factory=list)
    block: Optional[Block] = None  # stage parameters


@dataclass
class Arg(ASTNode):
    """Function argument: positional or named."""
    value: Expr
    name: Optional[str] = None  # if named argument


# Statements

@dataclass
class Decl(ASTNode):
    """Declaration: type ident = expr;"""
    typ: str  # 'target', 'corpus', 'tokenizer', 'model', 'evalset', 'unit'
    name: str
    expr: Expr


@dataclass
class Assign(ASTNode):
    """Assignment: ident = expr;"""
    name: str
    expr: Expr


@dataclass
class Assert(ASTNode):
    """Assertion: assert expr cmp expr;"""
    left: Expr
    op: str  # '<', '>', '<=', '>=', '=='
    right: Expr


@dataclass
class Export(ASTNode):
    """Export: export ident to string;"""
    name: str
    path: str


@dataclass
class Import(ASTNode):
    """Import: import string;"""
    path: str


@dataclass
class Block:
    """Block of key-value pairs: { key = value; ... }"""
    items: dict[str, Expr] = field(default_factory=dict)


# Program

@dataclass
class Program(ASTNode):
    """Root node: sequence of statements."""
    stmts: list[Stmt] = field(default_factory=list)


# Type aliases for convenience
Expr = Literal | Ident | BinOp | Member | Call
Stmt = Decl | Assign | Assert | Export | Import
