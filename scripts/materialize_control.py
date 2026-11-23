"""Regenerate or check the matched control submission."""

import argparse
import ast
from pathlib import Path
from tempfile import NamedTemporaryFile


def selector_assignments(tree: ast.Module) -> list[ast.AST]:
    def class_writes(node):
        selector = "_USE_WEIGHTED_VOTE"
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == selector:
                yield node
            return
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id == selector:
            yield node
        if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name == selector:
            yield node
        if isinstance(node, ast.MatchMapping) and node.rest == selector:
            yield node
        if isinstance(node, ast.alias) and (node.asname or node.name.split(".")[0]) == selector:
            yield node
        for child in ast.iter_child_nodes(node):
            yield from class_writes(child)

    return [assignment for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "WeightedVotePredictor"
            for statement in node.body for assignment in class_writes(statement)]


def derive_control(source: bytes, filename: str = "<candidate>") -> bytes:
    """Return the exact candidate bytes with the one vote selector disabled."""
    enabled = b"    _USE_WEIGHTED_VOTE = True"
    disabled = b"    _USE_WEIGHTED_VOTE = False"
    lines = source.splitlines(keepends=True)
    selectors = [index for index, line in enumerate(lines) if line.rstrip(b"\r\n") in (enabled, disabled)]
    assignments = selector_assignments(ast.parse(source, filename=filename))
    if (
        len(selectors) != 1 or lines[selectors[0]].rstrip(b"\r\n") != enabled
        or len(assignments) != 1 or assignments[0].lineno != selectors[0] + 1
    ):
        raise ValueError("The candidate must have exactly one enabled vote selector.")
    selector = selectors[0]
    lines[selector] = disabled + lines[selector][len(enabled):]
    return b"".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args()

    submissions = Path(__file__).resolve().parents[1] / "submissions"
    candidate = submissions / "weighted_vote.py"
    control = submissions / "weighted_vote_control.py"
    try:
        expected = derive_control(candidate.read_bytes(), str(candidate))
    except (SyntaxError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    if args.check:
        if not control.is_file() or control.read_bytes() != expected:
            parser.exit(1, "The control is missing or stale. Run scripts/materialize_control.py.\n")
        print("The matched control differs only at the weighted-vote selector.")
    else:
        staging = NamedTemporaryFile(dir=control.parent, prefix=f".{control.name}.", suffix=".tmp", delete=False)
        pending = Path(staging.name)
        try:
            with staging:
                staging.write(expected)
            pending.replace(control)
        finally:
            pending.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
