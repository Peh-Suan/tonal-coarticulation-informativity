"""Shared check: refuse to run on an input that is older than the file it was built from."""
import os, sys

def require_fresh(out, pairs):
    """pairs: [(input, must_be_newer_than), ...] (file names inside out/)."""
    bad = []
    for inp, upstream in pairs:
        a, b = os.path.join(out, inp), os.path.join(out, upstream)
        if not os.path.exists(a):
            bad.append(f"  {inp} is missing")
        elif os.path.exists(b) and os.path.getmtime(a) < os.path.getmtime(b):
            bad.append(f"  {inp} is OLDER than {upstream}: rerun the step that makes {inp} first")
    if bad and "--force" not in sys.argv:
        sys.exit("Stale or missing input (add --force to run anyway):\n" + "\n".join(bad))
    if "--force" in sys.argv: sys.argv.remove("--force")
