"""Verification harness (NOT part of the package): runs the entry's predict.py in-process under an audit hook that records every file open, network/socket event,
subprocess and third-party import. Usage: AUDIT_LOG=<json> python audit_run.py <predict.py> [args...]"""
import atexit, json, os, runpy, sys
LOG = os.environ["AUDIT_LOG"]; ev = {"open_read": set(), "open_write": set(), "network": [], "subprocess": [], "imports": set()}
def hook(name, args):
    if name == "open":
        path, mode, flags = args
        if isinstance(path, (str, bytes)):
            path = os.fsdecode(path)
            w = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
            (ev["open_write"] if w else ev["open_read"]).add(path)
    elif name.startswith("socket.") and name not in ("socket.gethostname", "socket.__new__"):
        ev["network"].append(name)
    elif name in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"):
        ev["subprocess"].append(str(args)[:200])
    elif name == "import":
        ev["imports"].add(args[0].split(".")[0])
sys.addaudithook(hook)
def dump():
    json.dump({k: (sorted(v) if isinstance(v, set) else v) for k, v in ev.items()}, open(LOG, "w"), indent=1)
atexit.register(dump)
target = sys.argv[1]; sys.argv = sys.argv[1:]; sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
runpy.run_path(target, run_name="__main__")
