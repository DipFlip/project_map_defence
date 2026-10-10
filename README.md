# Projected map defence

Tower defence projected onto a table. A USB camera tracks AprilTags (tag36h11,
print kit from DipFlip/ar-of-empires) placed on the table, and the projector
draws the game around them.

## Hardware / OS notes (Radxa Q6A)
- Projector on HDMI, rendered directly via DRM/KMS (no desktop needed). This needs
  the distro pygame (`python3-pygame`, SDL built with kmsdrm); the venv uses
  `--system-site-packages` for it. The pip pygame wheel has no kmsdrm support.
- Camera: Innomaker U20CAM (found automatically by name), 1920x1080 MJPG at 30 fps.
- `PMD_WINDOWED=1` gives a normal window instead of fullscreen KMS.

## Setup
    uv venv --system-site-packages .venv
    uv pip install --python .venv/bin/python opencv-contrib-python-headless "numpy<2.1" scipy
    sudo apt install python3-pygame v4l-utils

## Use
    .venv/bin/python tools/calibrate.py   # whenever camera/projector/table moves
    .venv/bin/python tools/preview.py     # rings around every tag seen
    .venv/bin/python run.py               # the game (Esc/q quits when a keyboard is attached)

## Layout
- `pmd/camera.py`       threaded capture, v4l2 exposure control
- `pmd/projector.py`    fullscreen KMS output
- `pmd/calibration.py`  projected ArUco patterns (pattern minus inverse capture to cancel
                        ambient light) -> joint fit of lens distortion + homography,
                        camera px <-> projector px. Saves `data/calibration.json`,
                        including `table_poly` (the area where the table plane was seen).
- `pmd/tags.py`, `pmd/tracker.py`  tag detection on a background thread, poses in projector px
- `pmd/game.py`         game loop skeleton; projector pixels are the world coordinates
