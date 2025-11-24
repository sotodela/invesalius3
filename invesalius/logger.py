import os, time, threading
from pathlib import Path
 
_log_lock = threading.Lock()
_log_file = None
 
def init_logger(node_name, base_dir="/logs"):
    """
    Creates a node-specific CSV file:
    /logs/timing_<node_name>_<timestamp>.csv
    """
    global _log_file
    Path(base_dir).mkdir(parents=True, exist_ok=True)
 
    ts = time.strftime("%Y%m%d-%H%M%S")
    filename = f"timing_{node_name}_{ts}.csv"
    path = os.path.join(base_dir, filename)
 
    _log_file = open(path, "a", buffering=1)  # line-buffered
    _log_file.write("seq,tag,t_ns\n")         # header once
    print(f"[{node_name}] timing log -> {path}")
 
def now_ns():
    return time.monotonic_ns()
 
def log_csv(seq, tag, t_ns=None):
    global _log_file
    if _log_file is None:
        return
    if t_ns is None:
        t_ns = now_ns()
    with _log_lock:
        _log_file.write(f"{seq},{tag},{t_ns}\n")