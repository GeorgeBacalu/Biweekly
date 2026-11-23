- source_spec: spec-2-walk-forward-evidence.md
  summary: Resolve the shipped runner's private temporary directory access in the managed Windows sandbox.
  evidence: Python 3.13's `tempfile.mkdtemp()` directory was inaccessible to the shipped loader in the managed attempt. The same offline run passed with approved temporary directory access.
