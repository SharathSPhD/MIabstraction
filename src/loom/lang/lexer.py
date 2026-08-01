"""Lexer for Loom language: tokenization with line/col tracking and unit-suffixed numbers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class TokenType(Enum):
    """Token types for Loom language."""
    # Literals
    IDENT = auto()
    NUMBER = auto()
    STRING = auto()

    # Keywords
    TARGET = auto()
    CORPUS = auto()
    TOKENIZER = auto()
    MODEL = auto()
    EVALSET = auto()
    UNIT = auto()
    ASSERT = auto()
    EXPORT = auto()
    IMPORT = auto()

    # Operators
    PLUS = auto()
    STAR = auto()
    EQUAL = auto()
    DOT = auto()
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()

    # Comparisons
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    EQ = auto()

    # Special
    EOF = auto()
    NEWLINE = auto()


KEYWORDS = {
    "target": TokenType.TARGET,
    "corpus": TokenType.CORPUS,
    "tokenizer": TokenType.TOKENIZER,
    "model": TokenType.MODEL,
    "evalset": TokenType.EVALSET,
    "unit": TokenType.UNIT,
    "assert": TokenType.ASSERT,
    "export": TokenType.EXPORT,
    "import": TokenType.IMPORT,
}


@dataclass
class SourcePos:
    """Source position: line, column (1-indexed)."""
    line: int
    col: int

    def __str__(self) -> str:
        return f"{self.line}:{self.col}"


@dataclass
class Token:
    """A token with type, value, and position."""
    type: TokenType
    value: str
    pos: SourcePos

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.pos})"


class Lexer:
    """Tokenizes Loom source code with line/col tracking."""

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: list[Token] = []

    def error(self, msg: str) -> None:
        """Raise a lexer error with position."""
        raise SyntaxError(f"Lexer error at {self.line}:{self.col}: {msg}")

    def peek(self, offset: int = 0) -> Optional[str]:
        """Peek at character without consuming."""
        p = self.pos + offset
        return self.source[p] if p < len(self.source) else None

    def advance(self, count: int = 1) -> str:
        """Consume and return characters."""
        result = ""
        for _ in range(count):
            if self.pos >= len(self.source):
                break
            ch = self.source[self.pos]
            result += ch
            self.pos += 1
            if ch == "\n":
                self.line += 1
                self.col = 1
            else:
                self.col += 1
        return result

    def skip_whitespace(self) -> None:
        """Skip whitespace and comments."""
        while True:
            # Skip whitespace
            while self.peek() and self.peek() in " \t\r\n":
                self.advance()

            # Skip comments
            if self.peek() == "/" and self.peek(1) == "/":
                while self.peek() and self.peek() != "\n":
                    self.advance()
                if self.peek() == "\n":
                    self.advance()
                # Loop back to skip any whitespace after the comment
            else:
                # No more whitespace or comments
                break

    def read_string(self) -> str:
        """Read a string literal."""
        assert self.peek() == '"'
        self.advance()  # consume opening "
        result = ""
        while self.peek() and self.peek() != '"':
            if self.peek() == "\\":
                self.advance()
                ch = self.peek()
                if ch == "n":
                    result += "\n"
                elif ch == "t":
                    result += "\t"
                elif ch == '"':
                    result += '"'
                elif ch == "\\":
                    result += "\\"
                else:
                    result += ch or ""
                self.advance()
            else:
                result += self.advance()
        if self.peek() != '"':
            self.error("Unterminated string literal")
        self.advance()  # consume closing "
        return result

    def read_number(self) -> float:
        """Read a number with optional unit suffix: 800M, 1.5B, 2%, 3e-4."""
        num_str = ""
        # Read digits, dots, and exponent
        while self.peek() and (self.peek().isdigit() or self.peek() in ".eE-+"):
            num_str += self.advance()

        # Parse base number
        try:
            value = float(num_str)
        except ValueError:
            self.error(f"Invalid number: {num_str}")

        # Handle unit suffix
        unit = self.peek()
        if unit and unit in "KMBT%":
            self.advance()
            if unit == "K":
                value *= 1e3
            elif unit == "M":
                value *= 1e6
            elif unit == "B":
                value *= 1e9
            elif unit == "T":
                value *= 1e12
            elif unit == "%":
                value /= 100.0

        return value

    def read_ident(self) -> str:
        """Read an identifier or keyword."""
        result = ""
        while self.peek() and (self.peek().isalnum() or self.peek() == "_"):
            result += self.advance()
        return result

    def tokenize(self) -> list[Token]:
        """Tokenize the entire source."""
        while self.pos < len(self.source):
            self.skip_whitespace()
            if self.pos >= len(self.source):
                break

            start_pos = SourcePos(self.line, self.col)
            ch = self.peek()

            if ch == '"':
                val = self.read_string()
                self.tokens.append(Token(TokenType.STRING, val, start_pos))
            elif ch.isdigit():
                val = self.read_number()
                self.tokens.append(Token(TokenType.NUMBER, str(val), start_pos))
            elif ch.isalpha() or ch == "_":
                ident = self.read_ident()
                if ident in KEYWORDS:
                    self.tokens.append(Token(KEYWORDS[ident], ident, start_pos))
                else:
                    self.tokens.append(Token(TokenType.IDENT, ident, start_pos))
            elif ch == "+":
                self.advance()
                self.tokens.append(Token(TokenType.PLUS, "+", start_pos))
            elif ch == "*":
                self.advance()
                self.tokens.append(Token(TokenType.STAR, "*", start_pos))
            elif ch == "=":
                self.advance()
                if self.peek() == "=":
                    self.advance()
                    self.tokens.append(Token(TokenType.EQ, "==", start_pos))
                else:
                    self.tokens.append(Token(TokenType.EQUAL, "=", start_pos))
            elif ch == "<":
                self.advance()
                if self.peek() == "=":
                    self.advance()
                    self.tokens.append(Token(TokenType.LE, "<=", start_pos))
                else:
                    self.tokens.append(Token(TokenType.LT, "<", start_pos))
            elif ch == ">":
                self.advance()
                if self.peek() == "=":
                    self.advance()
                    self.tokens.append(Token(TokenType.GE, ">=", start_pos))
                else:
                    self.tokens.append(Token(TokenType.GT, ">", start_pos))
            elif ch == ".":
                self.advance()
                self.tokens.append(Token(TokenType.DOT, ".", start_pos))
            elif ch == ",":
                self.advance()
                self.tokens.append(Token(TokenType.COMMA, ",", start_pos))
            elif ch == ";":
                self.advance()
                self.tokens.append(Token(TokenType.SEMICOLON, ";", start_pos))
            elif ch == "(":
                self.advance()
                self.tokens.append(Token(TokenType.LPAREN, "(", start_pos))
            elif ch == ")":
                self.advance()
                self.tokens.append(Token(TokenType.RPAREN, ")", start_pos))
            elif ch == "{":
                self.advance()
                self.tokens.append(Token(TokenType.LBRACE, "{", start_pos))
            elif ch == "}":
                self.advance()
                self.tokens.append(Token(TokenType.RBRACE, "}", start_pos))
            else:
                self.error(f"Unexpected character: {ch!r}")

        self.tokens.append(Token(TokenType.EOF, "", SourcePos(self.line, self.col)))
        return self.tokens
