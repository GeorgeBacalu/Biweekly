# Adversarial seam review

Verdict — tighten three boundaries before handoff. The single-file architecture and matched experiment remain viable.

## High · Preserve prediction row order

AD-4 can be read as a mapping from timestamps to vectors. A predictor stage can sort by time for smoothing and return the sorted frame with the correct timestamp labels, while the evaluation stage can rely on the kit’s positional handling. Both satisfy the stated mapping and count. The scorer replaces a nonmatching prediction index with the returns index without reordering values, so this pair can attach a signal to the wrong hour. Specify a pandas DataFrame in the exact incoming row order and index, and have local behavior checks compare it to the input. Asset labels must match the input ticker labels. [Architecture AD-4](../ARCHITECTURE-SPINE.md#ad-4--labeled-feature-and-position-contract-adopted) · [Kit evaluator](../../../../../requirements/evaluation.py#L57)

## High · Bind the readiness decision to one immutable candidate

AD-2 instructs developers to keep the evaluated bytes and AD-6 records artifact digests. The full-run producer and the novelty producer can still record different successful revisions, both as the candidate, while the decision consumer selects recent results by role and combines them. The evidence is individually attributable but no single file has cleared both gates. Require quick, full, gauge, model-behavior, and resource results used for readiness to refer to the same candidate digest and attempt snapshot. Bind the baseline and control by role and digest for the comparison, and make handoff select only the candidate digest named in the readiness result. [Architecture AD-5](../ARCHITECTURE-SPINE.md#ad-5--one-local-decision-authority-adopted) · [Architecture AD-6](../ARCHITECTURE-SPINE.md#ad-6--run-and-verdict-provenance)

## Medium · Verify map currency at the readiness decision

The environment rule stages and holds one snapshot for an attempt. The map producer can see any fresh acquisition as fulfilling AD-5, while a long comparison may finish after the official map publishes another snapshot. The consumer can then report readiness on a superseded distance. Define currency at the decision and withhold readiness until a superseding map is evaluated. Keep existing results attributable to the original map digest. [Architecture execution environments](../ARCHITECTURE-SPINE.md#execution-environments) · [Contest novelty rules](../../../../../requirements/COMPETITION.md#signal-cities-and-novelty)
