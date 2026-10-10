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
    .venv/bin/python tools/calibrate.py   # whenever camera/projector/table moves (tags spread flat, nobody leaning over)
    .venv/bin/python tools/table_frame.py # redo only the metric table frame
    .venv/bin/python tools/preview.py     # rings around every tag seen
    .venv/bin/python run.py               # menu + games (Esc/q quits when a keyboard is attached)
    .venv/bin/python tools/button_test.py # check the wand buttons

## Layout
- `pmd/camera.py`       threaded capture, v4l2 exposure control
- `pmd/projector.py`    fullscreen KMS output
- `pmd/calibration.py`  projected ArUco patterns (pattern minus inverse capture to cancel
                        ambient light) -> joint fit of lens distortion + homography,
                        camera px <-> projector px. Saves `data/calibration.json`,
                        including `table_poly` (the area where the table plane was seen).
- `pmd/table.py`        metric table frame: a homography projector px -> table mm fitted so every
                        flat tag is a square of its known size, aligned with the lit table outline.
                        Scenes draw on a canvas in table mm that is warped to the projector each frame,
                        so shapes are true on the table despite keystone.
- `pmd/tags.py`, `pmd/tracker.py`  tag detection on a background thread, poses in projector px
- `pmd/app.py`          scene manager: menu, games, wand button presses, back to menu
- `pmd/scenes/`         menu, snack castle, pong, pin shooter; they draw in table-canvas coordinates

## Wands and the menu
Tags G, B, D, C (16, 11, 13, 12) are handheld wands. Each has a red spot 2.9 tag
heights below the tag (towards the printed label); covering it is a button press
(the tracker counts red pixels around where the spot should be).
The games start at a menu: hold a wand over a game and press. Taking all four wands
off the table for 3 s goes back to the menu. Players: G+B is one pair, D+C the other.

## Pong
Players at the short ends. Your bouncer is the line between your two wands: hold them
closer for a faster return, tilt to aim. First to 7.

## Pin shooter
Ten pins in a line at each end. Each wand holds one ball; press to fire it straight
ahead (the tag's printed top). An empty wand catches an incoming ball, a loaded one
deflects it. Balls past your pin line roll into your ball return (corner on your
right, halfway to the centre); hold an empty wand there to pick one up. Knock down all ten to win.

## Snack castle: mice vs. the snacks
Mice run from the mouse hole along the path to steal snacks from the cheese castle.
Lose all 20 snacks and a new game starts after 8 s. Mice get stronger every 30 s.

| Tag | Piece | What it does |
|---|---|---|
| A (10) | coin collector | slide over coins dropped by caught mice, bring them to the chest |
| G (16) | chest | stores coins; park an attacker next to it for 1.5 s to upgrade (5, 10, 15, 20 coins) |
| C, E, J (12, 14, 19) | cat | swipes all mice around it, slow |
| B, D, F, H, I (11, 13, 15, 17, 18) | slingshot | shoots towards the tag's printed top (label text upright), 15° cone, one grid cell reach |

One grid cell is 40 mm, the spacing of the field-tag sheet.
