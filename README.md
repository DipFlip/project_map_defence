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
Tags G, B, D, C (16, 11, 13, 12) are handheld wands. Each has a blue spot 2.9 tag
heights below the tag (towards the printed label); covering it is a button press
(the tracker counts blue pixels around where the spot should be).
The games start at a menu: hold a wand over a game and press. Taking all four wands
off the table for 3 s goes back to the menu. Players: G+B is one pair, D+C the other.

## Pong
Players at the short ends. Your bouncer lies between your two wands (centred on them,
at most 70% of the court width). The ball bounces off it like off a moving wall: push
into the ball to hit it harder, sweep sideways to send it at an angle, tilt to aim.
Fast balls relax back to a calm cruising speed. First to 7.

## Webcam Pong (laptop, no projector)
The laptop screen is the table: the webcam view fills it (mirrored, so it works
like a mirror) and the Pong court is drawn over the video, 1:1 with the camera
image. Two players stand side by side in front of the laptop, each making a
finger gun (index finger out, thumb up, other fingers curled). Your bouncer is
the line between your thumb tip and index tip; the leftmost gun is the left
player, the rightmost the right player. Same ball physics as table Pong
(`pmd/scenes/hand_pong.py` subclasses it); the ball is served once both players
are aiming. Hand tracking is MediaPipe's hand landmarker (`pmd/hands.py`); its
model is downloaded to `data/` on first run.

    uv venv -p 3.12 .venv-cam
    uv pip install --python .venv-cam/bin/python -r requirements-webcam.txt
    .venv-cam/bin/python webcam_pong.py            # --windowed, --camera N, --no-mirror

Keys: Esc/q quit, r new match, v video brightness, d hand skeletons, f fullscreen.
On macOS the terminal needs camera access (System Settings > Privacy & Security > Camera).

### In the browser
`web/` is the same game as a static page (JS port of `hand_pong.py`, MediaPipe Tasks
Vision running in the page, nothing uploaded). Live at https://finger-gun-pong.vercel.app.

    python3 -m http.server 8517 -d web             # local: http://localhost:8517
    cd web && npx vercel deploy --prod             # redeploy from CLI

Or import the GitHub repo on vercel.com with default settings: `vercel.json` at the repo
root serves `web/` as a static site, no build step.

## Pin shooter
Ten pins in a line at each end. Each wand holds one ball; press to fire it wherever
the wand points (the tag's printed top). An empty wand catches an incoming ball, a loaded one
deflects it. Balls past your pin line roll into your ball return (both returns are on the
same side wall, each halfway between its player's end and the centre); hold an
empty wand in or near it to pick one up. Knock down all ten to win.

## Labyrinth
Four players, one wand each, standing at the corners. A shared tilting board in the
middle (half the table's short side wide); each wand's height (about 5 to 40 cm above
the table) sets its board corner's height. Roll the ball from the start through the
maze (avoid the holes), up the ramp onto the ledge, and jump the gap to the goal.
Falling resets to the last checkpoint. Drawn in perspective from above the board.
Wand heights come from how big the tag looks (`pmd/heights.py`); calibrate with
`tools/wand_height.py` (sweep each wand slowly between its lowest and highest point).

## Sneak
Single player. Put any wand on the green start spot; that wand is your agent. Creep
through the maze to the diamond and back without being seen: guards patrol with
flashlight cones (walls block them), and staying in one for half a second gets you
caught. Brushing a wall, or lifting the wand over one, makes a noise that brings
nearby guards running.

## Water
Tags A (10) and E (14) are taps on the far edge (the snack castle text side): turn
one from how it lay to open it (shut for the first 25° either way, fully open 90°
further; the dial shows where it points). A pours orange, E green; the
liquid flows towards the players. Every other tag is a tilting plank and each wand's
whole stick is a platform. Fill the containers with the colour they show (wrong
liquid fizzles away); full ones vanish and new ones appear.

## Slides
The menu's "slides" corner button starts a short talk on the table: the title slide,
the live camera view with what the tracker sees, the calibration capture, a game frame
next to the pre-skewed frame actually sent to the projector, and the closing slide
(`slides/*.png`). Any wand press moves on; after the last slide it's back to the menu.

## Sound and effects
`pmd/sound.py` synthesises all effects at start-up; `pmd/particles.py` draws bursts,
rings and confetti over every scene. Sound plays through the projector's speaker over
HDMI: that needs `firmware-qcom-audioreach` >= 1.0.4-3 (it ships the board's audio
topology, `QCS6490-Radxa-Dragon-Q6A-tplg.bin`); the app routes the HDMI output on start.

## Snack castle: mice vs. the snacks
Mice run from the mouse hole along the path to steal snacks from the cheese castle.
Lose all 20 snacks and a new game starts after 8 s. Mice get stronger every 30 s.

| Tag | Piece | What it does |
|---|---|---|
| B (11, wand) | coin collector | move over coins dropped by caught mice, bring them to the chest |
| G (16, wand) | chest | stores coins; park an attacker next to it for 1.5 s to upgrade (5, 10, 15, 20 coins) |
| E, J (14, 19) | cat | swipes all mice around it, slow |
| D, C (13, 12, wands) and A, F, H, I (10, 15, 17, 18) | slingshot | shoots towards the tag's printed top, 15° cone, two grid cells reach |

One grid cell is 40 mm, the spacing of the field-tag sheet.
