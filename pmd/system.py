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


def enable_hdmi_audio():
    """Route the board's playback stream to HDMI (the projector's speaker).

    The Radxa Dragon Q6A's sound card needs firmware-qcom-audioreach >= 1.0.4-3
    (it ships QCS6490-Radxa-Dragon-Q6A-tplg.bin). Its mixer starts with nothing
    routed to the DisplayPort/HDMI output; this switches that route on and returns
    the ALSA device for SDL, or None if there is no such card.
    """
    try:
        cards = open("/proc/asound/cards").read()
    except OSError:
        return None
    if "Dragon-Q6A" not in cards:
        return None
    card = cards.split("[")[0].strip().split()[0]
    subprocess.run(["amixer", "-q", "-c", card, "cset", "name=DISPLAY_PORT_RX_0 Audio Mixer MultiMedia1", "on"],
                   capture_output=True)
    return f"plughw:{card},0"
