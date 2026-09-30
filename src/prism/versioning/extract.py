"""Extract function-level snippets from JavaScript source with tree-sitter."""

from __future__ import annotations

from dataclasses import dataclass, field

import tree_sitter_javascript as tsjs
from tree_sitter import Language, Node, Parser

JS_EXTENSIONS = (".js", ".mjs", ".cjs", ".jsx")
MAX_FILE_SNIPPET_CHARS = 4000

_PARSER = Parser(Language(tsjs.language()))
_FUNC_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}


@dataclass(frozen=True)
class Snippet:
    path: str
    symbol: str
    start_line: int  # 1-based, inclusive
    end_line: int
    code: str
    callees: tuple[str, ...] = field(default=())


def _text(src: bytes, n: Node) -> str:
    return src[n.start_byte : n.end_byte].decode("utf-8", errors="replace")


def _callees(src: bytes, n: Node) -> tuple[str, ...]:
    out: list[str] = []
    stack = [n]
    while stack:
        cur = stack.pop()
        if cur.type == "call_expression":
            fn = cur.child_by_field_name("function")
            if fn is not None:
                if fn.type == "member_expression":
                    prop = fn.child_by_field_name("property")
                    fn = prop if prop is not None else fn
                out.append(_text(src, fn))
        stack.extend(cur.children)
    return tuple(sorted(set(out)))


def extract_js(path: str, source: str) -> list[Snippet]:
    src = source.encode("utf-8")
    tree = _PARSER.parse(src)
    snippets: list[Snippet] = []
    seen: dict[str, int] = {}

    def emit(name: str, node: Node) -> None:
        n = seen.get(name, 0)
        seen[name] = n + 1
        symbol = name if n == 0 else f"{name}#{n + 1}"
        snippets.append(
            Snippet(path, symbol, node.start_point[0] + 1, node.end_point[0] + 1,
                    _text(src, node), _callees(src, node))
        )

    def visit(node: Node, scope: str) -> None:
        t = node.type
        if t in ("function_declaration", "generator_function_declaration"):
            name = node.child_by_field_name("name")
            emit(f"{scope}{_text(src, name) if name else '<anonymous>'}", node)
            return
        if t == "class_declaration" or t == "class":
            name = node.child_by_field_name("name")
            cls = _text(src, name) if name else "<class>"
            body = node.child_by_field_name("body")
            for ch in body.children if body else []:
                visit(ch, f"{scope}{cls}.")
            return
        if t == "method_definition":
            name = node.child_by_field_name("name")
            emit(f"{scope}{_text(src, name) if name else '<method>'}", node)
            return
        if t == "variable_declarator":
            value = node.child_by_field_name("value")
            name = node.child_by_field_name("name")
            if value is not None and value.type in _FUNC_VALUES and name is not None:
                # include the `const x =` so the snippet reads like source
                parent = node.parent if node.parent is not None else node
                emit(f"{scope}{_text(src, name)}", parent)
                return
            if value is not None and value.type == "class":
                visit(value, scope)
                return
        for ch in node.children:
            visit(ch, scope)

    visit(tree.root_node, "")
    if not snippets and source.strip():
        # module with no functions (config, constants): keep one file-level snippet
        lines = source.count("\n") + 1
        snippets.append(Snippet(path, "<module>", 1, lines, source[:MAX_FILE_SNIPPET_CHARS]))
    return snippets
