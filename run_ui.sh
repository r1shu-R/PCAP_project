#!/usr/bin/env bash
# Build (if needed) and start the UI. Run inside WSL from the project folder:
#   ./run_ui.sh
# then open http://localhost:8501 in your Windows browser.
set -e
cd "$(dirname "$0")"

export PATH="/usr/local/cuda/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1          # keep __pycache__ out of OneDrive
VENV="$HOME/.venvs/pcap"

if [ ! -x "$VENV/bin/streamlit" ]; then
    echo "Setting up Python environment in $VENV ..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q -r requirements.txt
fi

make
echo
echo "  Open http://localhost:8501 in your browser (Ctrl+C here to stop)."
echo
exec "$VENV/bin/streamlit" run ui/app.py --server.headless true --browser.gatherUsageStats false
