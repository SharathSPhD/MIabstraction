"""Parser for Loom language: recursive descent implementing EBNF grammar."""
from __future__ import annotations

from typing import Optional

from .lexer import Token, TokenType, Lexer, SourcePos
from . import ast


class Parser:
    """Recursive descent parser for Loom language."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    @classmethod
    def from_source(cls, source: str) -> Parser:
        """Create parser from source code string."""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        return cls(tokens)

    def error(self, msg: str) -> None:
        """Raise a parse error with context."""
        tok = self.current()
        raise SyntaxError(
            f"Parse error at {tok.pos}: {msg}\n"
            f"  Found: {tok.type.name} {tok.value!r}\n"
            f"  Expected: {msg}"
        )

    def current(self) -> Token:
        """Get current token."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # EOF

    def peek(self, offset: int = 0) -> Token:
        """Peek at token."""
        p = self.pos + offset
        if p < len(self.tokens):
            return self.tokens[p]
        return self.tokens[-1]

    def advance(self) -> Token:
        """Consume and return current token."""
        tok = self.current()
        if tok.type != TokenType.EOF:
            self.pos += 1
        return tok

    def expect(self, typ: TokenType) -> Token:
        """Expect a specific token type."""
        tok = self.current()
        if tok.type != typ:
            self.error(f"Expected {typ.name}, got {tok.type.name}")
        return self.advance()

    def match(self, *types: TokenType) -> bool:
        """Check if current token matches any type."""
        return self.current().type in types

    def consume(self, typ: TokenType) -> Token:
        """Consume if matches, else error."""
        if not self.match(typ):
            self.error(f"Expected {typ.name}")
        return self.advance()

    def parse(self) -> ast.Program:
        """Parse a program."""
        stmts = []
        pos = self.current().pos
        while not self.match(TokenType.EOF):
            stmt = self.parse_stmt()
            if stmt:
                stmts.append(stmt)
        return ast.Program(stmts=stmts, pos=pos)

    def parse_stmt(self) -> Optional[ast.Stmt]:
        """Parse a statement."""
        pos = self.current().pos

        # Type declaration: type ident = expr;
        if self.match(
            TokenType.TARGET,
            TokenType.CORPUS,
            TokenType.TOKENIZER,
            TokenType.MODEL,
            TokenType.EVALSET,
            TokenType.UNIT,
        ):
            typ_tok = self.advance()
            typ = typ_tok.value
            name_tok = self.expect(TokenType.IDENT)
            name = name_tok.value
            self.expect(TokenType.EQUAL)
            expr = self.parse_expr()
            self.expect(TokenType.SEMICOLON)
            return ast.Decl(typ=typ, name=name, expr=expr, pos=pos)

        # Assert: assert expr cmp expr;
        if self.match(TokenType.ASSERT):
            self.advance()
            left = self.parse_expr()
            if not self.match(TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE, TokenType.EQ):
                self.error("Expected comparison operator after assert")
            op_tok = self.advance()
            op = op_tok.value
            right = self.parse_expr()
            self.expect(TokenType.SEMICOLON)
            return ast.Assert(left=left, op=op, right=right, pos=pos)

        # Export: export ident to string;
        if self.match(TokenType.EXPORT):
            self.advance()
            name_tok = self.expect(TokenType.IDENT)
            name = name_tok.value
            self.expect(TokenType.IDENT)  # "to" is an identifier, not a keyword
            if self.tokens[self.pos - 1].value != "to":
                self.error("Expected 'to' in export statement")
            str_tok = self.expect(TokenType.STRING)
            path = str_tok.value
            self.expect(TokenType.SEMICOLON)
            return ast.Export(name=name, path=path, pos=pos)

        # Import: import string;
        if self.match(TokenType.IMPORT):
            self.advance()
            str_tok = self.expect(TokenType.STRING)
            path = str_tok.value
            self.expect(TokenType.SEMICOLON)
            return ast.Import(path=path, pos=pos)

        # Assignment: ident = expr;
        if self.match(TokenType.IDENT):
            # Look ahead to distinguish decl from assign
            if self.peek(1).type == TokenType.EQUAL and not self.match(
                TokenType.TARGET,
                TokenType.CORPUS,
                TokenType.TOKENIZER,
                TokenType.MODEL,
                TokenType.EVALSET,
                TokenType.UNIT,
            ):
                name_tok = self.advance()
                name = name_tok.value
                self.expect(TokenType.EQUAL)
                expr = self.parse_expr()
                self.expect(TokenType.SEMICOLON)
                return ast.Assign(name=name, expr=expr, pos=pos)

        # If we get here and didn't parse anything, it's an error
        if not self.match(TokenType.EOF):
            self.error(f"Unexpected token in statement: {self.current().type.name}")
        return None

    def parse_expr(self) -> ast.Expr:
        """Parse an expression (handles binop lowest precedence)."""
        return self.parse_binop()

    def parse_binop(self) -> ast.Expr:
        """Parse binary operations: + and * (left-associative, same precedence for now)."""
        left = self.parse_member()
        pos = left.pos

        while self.match(TokenType.PLUS, TokenType.STAR):
            op_tok = self.advance()
            op = op_tok.value
            right = self.parse_member()
            left = ast.BinOp(op=op, left=left, right=right, pos=pos)

        return left

    def parse_member(self) -> ast.Expr:
        """Parse member access and method calls."""
        left = self.parse_call()

        while self.match(TokenType.DOT):
            self.advance()
            member_tok = self.expect(TokenType.IDENT)
            member = member_tok.value
            pos = member_tok.pos

            # Check if method call
            if self.match(TokenType.LPAREN):
                args = self.parse_args()
                left = ast.Member(object=left, member=member, args=args, pos=pos)
            else:
                left = ast.Member(object=left, member=member, pos=pos)

        return left

    def parse_call(self) -> ast.Expr:
        """Parse a primary expression (literals, identifiers, calls, parenthesized)."""
        pos = self.current().pos

        # Literal: number or string
        if self.match(TokenType.NUMBER):
            tok = self.advance()
            value = float(tok.value)
            return ast.Literal(value=value, pos=pos)

        if self.match(TokenType.STRING):
            tok = self.advance()
            value = tok.value
            return ast.Literal(value=value, pos=pos)

        # Identifier or path.call (allow keywords that can be module/function names)
        if self.match(TokenType.IDENT) or self._is_path_keyword():
            path = []
            while self.match(TokenType.IDENT) or self._is_path_keyword():
                tok = self.advance()
                path.append(tok.value)
                if self.match(TokenType.DOT):
                    self.advance()
                else:
                    break

            # If this is a call (followed by paren), parse as call
            if self.match(TokenType.LPAREN):
                args = self.parse_args()
                block = None
                if self.match(TokenType.LBRACE):
                    block = self.parse_block()
                return ast.Call(path=path, args=args, block=block, pos=pos)

            # Otherwise, return as identifier
            if len(path) == 1:
                return ast.Ident(name=path[0], pos=pos)
            else:
                # Multi-part path without call is member access
                result = ast.Ident(name=path[0], pos=pos)
                for part in path[1:]:
                    result = ast.Member(object=result, member=part, pos=pos)
                return result

        # Parenthesized expression
        if self.match(TokenType.LPAREN):
            self.advance()
            expr = self.parse_expr()
            self.expect(TokenType.RPAREN)
            return expr

        self.error(f"Expected expression, got {self.current().type.name}")

    def _is_path_keyword(self) -> bool:
        """Check if current token is a keyword that can be used in paths."""
        return self.match(TokenType.TOKENIZER, TokenType.TARGET, TokenType.CORPUS, TokenType.MODEL)

    def parse_args(self) -> list[ast.Arg]:
        """Parse function arguments: ( [arg [, arg]* ] )"""
        self.expect(TokenType.LPAREN)
        args = []

        if not self.match(TokenType.RPAREN):
            while True:
                pos = self.current().pos
                # Check for named argument
                if self.match(TokenType.IDENT) and self.peek(1).type == TokenType.EQUAL:
                    name_tok = self.advance()
                    name = name_tok.value
                    self.advance()  # consume =
                    value = self.parse_expr()
                    args.append(ast.Arg(value=value, name=name, pos=pos))
                else:
                    value = self.parse_expr()
                    args.append(ast.Arg(value=value, pos=pos))

                if not self.match(TokenType.COMMA):
                    break
                self.advance()

        self.expect(TokenType.RPAREN)
        return args

    def parse_block(self) -> ast.Block:
        """Parse a block: { ident = expr; ... }"""
        self.expect(TokenType.LBRACE)
        items = {}

        while not self.match(TokenType.RBRACE):
            key_tok = self.expect(TokenType.IDENT)
            key = key_tok.value
            self.expect(TokenType.EQUAL)
            value = self.parse_expr()
            self.expect(TokenType.SEMICOLON)
            items[key] = value

        self.expect(TokenType.RBRACE)
        return ast.Block(items=items)
