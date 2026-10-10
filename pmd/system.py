"""Host tweaks for low latency on the Radxa."""
import contextlib
import glob
import subprocess


@contextlib.contextmanager
def performance_governor():
    """Pin CPU frequency to max while the game runs.

    With the default schedutil governor the cores clock down while waiting for
    each camera frame, which made tag detection ~4x slower. Needs passwordless
    sudo; otherwise it just warns and carries on.
    """
    paths = sorted(glob.glob("/sys/devices/system/cpu/cpufreq/policy*/scaling_governor"))
    before = {p: open(p).read().strip() for p in paths}

    def write(p, value):
        return subprocess.run(["sudo", "-n", "tee", p], input=value.encode(), capture_output=True).returncode == 0

    ok = all(before[p] == "performance" or write(p, "performance") for p in paths)
    if not ok:
        print("warning: could not set the performance CPU governor; tracking will lag more")
    try:
        yield
    finally:
        for p, gov in before.items():
            if gov != "performance":
                write(p, gov)
