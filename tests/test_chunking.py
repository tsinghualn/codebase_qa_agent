import ast
import textwrap

from codebase_qa_agent.chunking import chunk_python_file


SAMPLE = textwrap.dedent(
    '''
    def foo(x):
        return x + 1

    class Bar:
        def method(self):
            return 42
    '''
)


def test_chunking_aligns_with_ast_function_boundaries():
    chunks = chunk_python_file("sample.py", SAMPLE)
    assert chunks
    tree = ast.parse(SAMPLE)
    symbols = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    chunk_symbols = {c.symbol for c in chunks}
    assert symbols <= chunk_symbols
    for chunk in chunks:
        if chunk.symbol == "foo":
            assert chunk.start_line == 2
            assert "return x + 1" in chunk.text