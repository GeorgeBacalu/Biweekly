"""Regenerate or check the matched control submission."""

import argparse
import ast
from pathlib import Path
from tempfile import NamedTemporaryFile


def selector_assignments(tree: ast.Module) -> list[ast.stmt]:
    assignments = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "WeightedVotePredictor":
            continue
        for statement in node.body:
            if isinstance(statement, ast.Assign):
                targets = statement.targets
            elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
                targets = [statement.target]
            else:
                continue
            if any(
                isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)
                and name.id == "_USE_WEIGHTED_VOTE"
                for target in targets for name in ast.walk(target)
            ):
                assignments.append(statement)
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args()

    submissions = Path(__file__).resolve().parents[1] / "submissions"
    candidate = submissions / "weighted_vote.py"
    control = submissions / "weighted_vote_control.py"
    source = candidate.read_bytes()
    enabled = b"    _USE_WEIGHTED_VOTE = True"
    disabled = b"    _USE_WEIGHTED_VOTE = False"
    lines = source.splitlines(keepends=True)
    selectors = [index for index, line in enumerate(lines) if line.rstrip(b"\r\n") in (enabled, disabled)]
    assignments = selector_assignments(ast.parse(source, filename=str(candidate)))
    if (
        len(selectors) != 1 or lines[selectors[0]].rstrip(b"\r\n") != enabled
        or len(assignments) != 1 or assignments[0].lineno != selectors[0] + 1
    ):
        parser.exit(1, "The candidate must have exactly one enabled vote selector.\n")
    selector = selectors[0]
    lines[selector] = disabled + lines[selector][len(enabled):]
    expected = b"".join(lines)

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
