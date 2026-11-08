"""Proves the copied Thanos files are byte-identical to Kareem's pinned commit (origin/main 8e00a50): compares git blob ids."""
import subprocess, sys, os
C = sys.argv[1] if len(sys.argv) > 1 else "8e00a50"; root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip(); ok = True
for src, dst in (("house-edge/thanos/__init__.py", "thanos/__init__.py"), ("house-edge/thanos/egp.py", "thanos/egp.py"), ("house-edge/thanos/model.py", "thanos/model.py"),
                 ("house-edge/common/__init__.py", "common/__init__.py"), ("house-edge/common/data.py", "common/data.py")):
    want = subprocess.check_output(["git", "-C", root, "rev-parse", f"{C}:{src}"], text=True).strip(); got = subprocess.check_output(["git", "hash-object", os.path.join("deploy_thanos/entry_src", dst)], text=True).strip()
    print(f"{dst:22s} {'IDENTICAL' if want == got else 'DIFFERS'}  blob {got[:10]}"); ok &= want == got
print("ALL PINNED FILES IDENTICAL TO", C if ok else "-- MISMATCH"); sys.exit(0 if ok else 1)
