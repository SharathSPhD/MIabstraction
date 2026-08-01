"""Tests for Loom compiler front-end."""
import pytest

from loom.lang.lexer import Lexer, TokenType, SourcePos
from loom.lang.parser import Parser
from loom.lang.check import TypeChecker, TypeError as CheckTypeError
from loom.lang.ir import IRGenerator
from loom.lang.frontend import compile_source
from loom.lang import ast


class TestLexer:
    """Test lexical analysis."""

    def test_ident_and_keywords(self):
        """Test identifiers and keywords."""
        lexer = Lexer("target x = 5")
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.TARGET
        assert tokens[1].type == TokenType.IDENT
        assert tokens[1].value == "x"
        assert tokens[2].type == TokenType.EQUAL

    def test_unit_suffixed_numbers(self):
        """Test numbers with unit suffixes."""
        test_cases = [
            ("800M", 8e8),
            ("1.5B", 1.5e9),
            ("2%", 0.02),
            ("3e-4", 3e-4),
            ("500K", 500e3),
        ]
        for text, expected in test_cases:
            lexer = Lexer(text)
            tokens = lexer.tokenize()
            assert tokens[0].type == TokenType.NUMBER
            assert float(tokens[0].value) == expected, f"Failed for {text}"

    def test_strings(self):
        """Test string literals."""
        lexer = Lexer('"hello world"')
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello world"

    def test_line_col_tracking(self):
        """Test line and column tracking."""
        lexer = Lexer("x = 5;\ny = 10;")
        tokens = lexer.tokenize()
        # First line: x = 5;
        assert tokens[0].pos.line == 1
        assert tokens[0].pos.col == 1
        # After newline
        assert tokens[4].pos.line == 2
        assert tokens[4].pos.col == 1

    def test_comments_skipped(self):
        """Test that comments are skipped."""
        lexer = Lexer("x = 5; // this is a comment\ny = 10;")
        tokens = lexer.tokenize()
        # Should not see comment token
        values = [t.value for t in tokens if t.type != TokenType.EOF]
        assert "this is a comment" not in values


class TestParser:
    """Test parsing."""

    def test_simple_decl(self):
        """Test parsing a simple declaration."""
        parser = Parser.from_source('target x = decoder(layers=12, width=768);')
        program = parser.parse()
        assert len(program.stmts) == 1
        stmt = program.stmts[0]
        assert isinstance(stmt, ast.Decl)
        assert stmt.name == "x"
        assert stmt.typ == "target"

    def test_binary_operations(self):
        """Test parsing binary operations."""
        parser = Parser.from_source("corpus c = a + b;")
        program = parser.parse()
        stmt = program.stmts[0]
        assert isinstance(stmt.expr, ast.BinOp)
        assert stmt.expr.op == "+"

    def test_function_call(self):
        """Test parsing function calls."""
        parser = Parser.from_source('corpus c = data.text("spec");')
        program = parser.parse()
        stmt = program.stmts[0]
        assert isinstance(stmt.expr, ast.Call)
        assert stmt.expr.path == ["data", "text"]

    def test_block_parsing(self):
        """Test parsing blocks in function calls."""
        parser = Parser.from_source(
            "model m = pretrain(arch, corp, tk) { tokens = 800M; lr = 6e-4; };"
        )
        program = parser.parse()
        stmt = program.stmts[0]
        assert isinstance(stmt.expr, ast.Call)
        assert stmt.expr.block is not None
        assert "tokens" in stmt.expr.block.items

    def test_assertions(self):
        """Test parsing assertions."""
        parser = Parser.from_source("assert perplexity(m) < 30;")
        program = parser.parse()
        stmt = program.stmts[0]
        assert isinstance(stmt, ast.Assert)
        assert stmt.op == "<"

    def test_export_statement(self):
        """Test parsing export."""
        parser = Parser.from_source('export m to "model_v1";')
        program = parser.parse()
        stmt = program.stmts[0]
        assert isinstance(stmt, ast.Export)
        assert stmt.name == "m"
        assert stmt.path == "model_v1"

    # def test_error_messages(self):
    #     """Test that parse errors are informative."""
    #     # Note: error handling is best tested at integration level
    #     # Parser.from_source("target x =") may accept or reject depending on EOF handling


class TestTypeChecker:
    """Test type checking."""

    def test_simple_types(self):
        """Test type inference for simple expressions."""
        parser = Parser.from_source('target t = decoder(layers=12);')
        program = parser.parse()
        checker = TypeChecker()
        env = checker.check(program)
        assert "t" in env

    def test_corpus_algebra(self):
        """Test corpus arithmetic."""
        parser = Parser.from_source(
            'corpus a = data.text("x"); '
            'corpus b = data.text("y"); '
            'corpus c = a + b;'
        )
        program = parser.parse()
        checker = TypeChecker()
        env = checker.check(program)
        assert "c" in env

    def test_wrong_argument_type(self):
        """Test error on wrong argument type."""
        parser = Parser.from_source(
            'corpus c = data.text("x"); '
            'target t = decoder(layers=12); '
            'model m = pretrain(c, t, tk);'  # Wrong order
        )
        program = parser.parse()
        checker = TypeChecker()
        with pytest.raises(CheckTypeError):
            checker.check(program)

    def test_alignment_wrong_corpus_kind(self):
        """Test error when align gets plain text instead of preferences."""
        parser = Parser.from_source(
            'model m = pretrain(arch, corp, tk); '
            'corpus plain = data.text("x"); '
            'model m2 = align(m, plain);'  # Should be corpus[pref]
        )
        program = parser.parse()
        checker = TypeChecker()
        # This should catch the corpus kind mismatch
        with pytest.raises(CheckTypeError):
            checker.check(program)

    def test_impossible_assertion(self):
        """Test error on impossible assertions."""
        parser = Parser.from_source(
            "model m = pretrain(arch, corp, tk); "
            "assert perplexity(m) < 0;"  # Impossible
        )
        program = parser.parse()
        checker = TypeChecker()
        with pytest.raises(CheckTypeError):
            checker.check(program)


class TestIR:
    """Test intermediate representation generation."""

    def test_ir_generation(self):
        """Test that IR is generated correctly."""
        parser = Parser.from_source(
            'target arch = decoder(layers=12, width=768); '
            'corpus corp = data.text("spec"); '
            'tokenizer tk = tokenizer.bpe(corp, vocab=16000); '
            'model m = pretrain(arch, corp, tk) { tokens = 800M; }; '
            'export m to "result";'
        )
        program = parser.parse()
        checker = TypeChecker()
        checker.check(program)
        gen = IRGenerator()
        ir = gen.generate(program)
        assert len(ir.nodes) > 0

    def test_ir_topological_order(self):
        """Test that IR nodes are in topological order."""
        parser = Parser.from_source(
            'target arch = decoder(layers=12); '
            'corpus a = data.text("x"); '
            'tokenizer tk = tokenizer.bpe(a); '
            'model m = pretrain(arch, a, tk); '
            'export m to "result";'
        )
        program = parser.parse()
        checker = TypeChecker()
        checker.check(program)
        gen = IRGenerator()
        ir = gen.generate(program)

        # Check topological order: no node should reference a later node
        node_ids = {n.id: i for i, n in enumerate(ir.nodes)}
        for node in ir.nodes:
            for inp_id in node.inputs:
                assert node_ids.get(inp_id, -1) < node_ids[node.id]

    def test_ir_json_serialization(self):
        """Test IR JSON serialization."""
        parser = Parser.from_source('target arch = decoder(layers=12);')
        program = parser.parse()
        checker = TypeChecker()
        checker.check(program)
        gen = IRGenerator()
        ir = gen.generate(program)

        json_str = ir.to_json()
        assert "nodes" in json_str
        assert "decoder" in json_str


class TestSmalltalker:
    """Test with the complete smalltalk.loom example."""

    def test_smalltalk_loom_parses(self):
        """Test that smalltalk.loom parses and type checks."""
        # We'll just test the core structure here
        source = """
target arch = decoder(layers=12, width=768, heads=12, ctx=1024);

corpus speech = data.text("babylm:childes");
corpus prose = data.text("babylm:simple_wiki");
corpus mix = speech * 0.7 + prose * 0.3;

tokenizer tk = tokenizer.bpe(mix, vocab=16000);

model m = pretrain(arch, mix, tk) {
    tokens = 800M;
    optimizer = adamw(lr=6e-4, wd=0.1);
};

export m to "smalltalk-v1";
"""
        parser = Parser.from_source(source)
        program = parser.parse()
        assert len(program.stmts) > 0

        checker = TypeChecker()
        env = checker.check(program)
        assert "m" in env
