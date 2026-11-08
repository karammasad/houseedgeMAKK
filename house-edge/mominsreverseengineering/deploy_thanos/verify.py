"""Clean-folder deployment verification of the Thanos E2 entry (organizer-like conditions).  Run from mominsreverseengineering/:
    python3 deploy_thanos/verify.py [--data ../data/main] [--runs 2]
Steps: package with the kit's own packager (validates ZIP contract) -> extract with the kit's extractor into a NEW empty folder -> make the code folder read-only ->
run `python predict.py --data <data> --out <out>/submission.csv` from that folder with an EMPTY environment, OS-level network denial (sandbox-exec), threads pinned to 2,
an audit hook recording every file opened / socket / subprocess / import -> /usr/bin/time -l for wall, CPU and peak RSS -> kit check_submission.py -> sha256 and a
second independent run for byte-level determinism."""
import argparse, hashlib, json, os, re, shutil, stat, subprocess, sys, tempfile, time
ROOT = os.path.abspath("."); KIT = os.path.abspath("../kit"); sys.path.insert(0, KIT)
ap = argparse.ArgumentParser(); ap.add_argument("--data", default=os.path.abspath("../data/main")); ap.add_argument("--runs", type=int, default=2); A = ap.parse_args()
from slot_forecasting.submission_contract import package_code, extract_archive
work = tempfile.mkdtemp(prefix="thanos_verify_", dir="/private/tmp/claude-501/-Users-mominhassan-houseedgeMAKK/849ad2ca-03cb-465f-9462-bba974b3e3d3/scratchpad")
zip_path = os.path.join(ROOT, "deploy_thanos", "entry.zip")
if os.path.exists(zip_path): os.remove(zip_path)
package_code(os.path.join(ROOT, "deploy_thanos", "entry_src"), zip_path); print("ZIP:", zip_path, os.path.getsize(zip_path), "bytes")
ext = os.path.join(work, "submission"); cfg = extract_archive(open(zip_path, "rb").read(), ext); print("reproduce.json:", cfg)
for dp, dn, fn in os.walk(ext):
    for f in fn: os.chmod(os.path.join(dp, f), 0o444)
    os.chmod(dp, 0o555)                                                   # /submission is read-only in the organizer run
files = sorted(os.path.relpath(os.path.join(dp, f), ext) for dp, _, fn in os.walk(ext) for f in fn); print("extracted files:", files)
PROFILE = "(version 1)(allow default)(deny network*)"; py = sys.executable
net = subprocess.run(["sandbox-exec", "-p", PROFILE, py, "-c", "import urllib.request;urllib.request.urlopen('http://1.1.1.1',timeout=4)"], capture_output=True, text=True)
print("network sanity under the sandbox profile (must FAIL):", "blocked" if net.returncode != 0 else "NOT BLOCKED", "|", (net.stderr.strip().splitlines() or [""])[-1][:100])
res = []
for r in range(A.runs):
    out = os.path.join(work, f"output{r}"); os.makedirs(out); tmp = os.path.join(work, f"tmp{r}"); os.makedirs(tmp); log = os.path.join(work, f"audit{r}.json")
    cmd = ["/usr/bin/time", "-l", "sandbox-exec", "-p", PROFILE, "/usr/bin/env", "-i", "PATH=/usr/bin:/bin", f"HOME={tmp}", f"TMPDIR={tmp}", "PYTHONDONTWRITEBYTECODE=1", f"AUDIT_LOG={log}",
           py, "-B", os.path.join(ROOT, "deploy_thanos", "audit_run.py"), os.path.join(ext, "predict.py"), "--data", A.data, "--out", os.path.join(out, "submission.csv")]
    if r > 0:                                                              # independent second run: plain predict.py, no audit harness
        cmd = ["/usr/bin/time", "-l", "sandbox-exec", "-p", PROFILE, "/usr/bin/env", "-i", "PATH=/usr/bin:/bin", f"HOME={tmp}", f"TMPDIR={tmp}", "PYTHONDONTWRITEBYTECODE=1",
               py, "-B", os.path.join(ext, "predict.py"), "--data", A.data, "--out", os.path.join(out, "submission.csv")]
    t0 = time.time(); p = subprocess.run(cmd, cwd=ext, capture_output=True, text=True); wall = time.time() - t0
    m = lambda pat: float(re.search(pat, p.stderr).group(1)) if re.search(pat, p.stderr) else None
    rss = m(r"(\d+)\s+maximum resident set size"); user = m(r"([\d.]+)\s+user"); sysc = m(r"([\d.]+)\s+sys")
    csvp = os.path.join(out, "submission.csv"); ok = p.returncode == 0 and os.path.exists(csvp)
    row = dict(run=r, returncode=p.returncode, wall_s=round(wall, 1), user_s=user, sys_s=sysc, cpu_s=round((user or 0) + (sysc or 0), 1), peak_rss_mb=round((rss or 0) / 1e6, 1),
               cores_equiv=round(((user or 0) + (sysc or 0)) / wall, 2), csv=csvp if ok else None, sha256=hashlib.sha256(open(csvp, "rb").read()).hexdigest() if ok else None,
               csv_bytes=os.path.getsize(csvp) if ok else None, stdout_tail=p.stdout.strip()[-120:], stderr_tail="" if ok else p.stderr[-600:])
    if r == 0 and os.path.exists(log):
        a = json.load(open(log)); data_reads = sorted(x for x in a["open_read"] if A.data in x); other = sorted(os.path.basename(x) for x in data_reads)
        row["audit"] = dict(data_files_read=sorted({os.path.relpath(x, A.data) for x in data_reads}), network_events=a["network"], subprocess=a["subprocess"],
                            writes_outside_out_tmp=sorted(x for x in a["open_write"] if not (x.startswith(out) or x.startswith(tmp) or x.startswith("/dev"))),
                            third_party_imports=sorted(i for i in a["imports"] if i in ("numpy", "scipy", "sklearn", "pandas", "torch", "joblib", "requests", "urllib3", "socket")))
    res.append(row); print(json.dumps(row, indent=1)); sys.stdout.flush()
if len(res) > 1 and all(x["sha256"] for x in res): print("BYTE-IDENTICAL across runs:", len({x["sha256"] for x in res}) == 1)
chk = subprocess.run([py, os.path.join(KIT, "check_submission.py"), res[0]["csv"], "--data", A.data], capture_output=True, text=True, cwd=KIT); print("check_submission rc", chk.returncode); print((chk.stdout + chk.stderr).strip()[-700:])
json.dump(dict(runs=res, work=work, checker_rc=chk.returncode, checker=(chk.stdout + chk.stderr).strip()[-700:]), open(os.path.join(ROOT, "deploy_thanos", "verify_result.json"), "w"), indent=1)
