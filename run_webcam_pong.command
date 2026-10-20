#!/bin/bash
# Double-click in Finder to set up (first time) and start webcam Pong.
cd "$(dirname "$0")"
set -e
if [ ! -x .venv-cam/bin/python ]; then
  echo "== first run: creating .venv-cam =="
  if command -v uv >/dev/null; then
    uv venv -p 3.12 .venv-cam
    uv pip install --python .venv-cam/bin/python -r requirements-webcam.txt
  else
    PY=""
    for v in python3.12 python3.11 python3.13 python3.10 python3; do
      if command -v $v >/dev/null; then PY=$v; break; fi
    done
    echo "using $PY ($($PY --version))"
    $PY -m venv .venv-cam
    .venv-cam/bin/python -m pip install --upgrade pip
    .venv-cam/bin/python -m pip install -r requirements-webcam.txt
  fi
fi
set +e
echo "== starting webcam pong (Esc to quit) =="
.venv-cam/bin/python webcam_pong.py "$@"
echo "== exited; press Enter to close =="
read
