#!/usr/bin/env bash
# PartField Studio — Modal Setup Script
# Run this once on your local machine to deploy everything.
set -e

echo ""
echo "═══════════════════════════════════════════════"
echo "  PartField Studio — Modal Deployment Setup"
echo "═══════════════════════════════════════════════"
echo ""

# ── 1. Check dependencies ──────────────────────────────────────────────────
echo "▶ Checking dependencies…"

if ! command -v python3 &>/dev/null; then
  echo "  ✗ Python 3 not found. Install from https://python.org"
  exit 1
fi

if ! command -v modal &>/dev/null; then
  echo "  Installing Modal CLI…"
  pip install modal
fi

echo "  ✓ Modal CLI ready"
echo ""

# ── 2. Modal auth ──────────────────────────────────────────────────────────
echo "▶ Authenticating with Modal…"
echo "  (A browser window will open — log in / sign up at modal.com)"
echo ""
modal setup
echo ""

# ── 3. Copy web app files ──────────────────────────────────────────────────
echo "▶ Copying web app files…"

# These should already exist alongside this script if you followed instructions
if [ ! -f "modal_app.py" ]; then
  echo "  ✗ modal_app.py not found. Make sure you're in the project directory."
  exit 1
fi

echo "  ✓ Files ready"
echo ""

# ── 4. Download model weights into Modal Volume ────────────────────────────
echo "▶ Downloading PartField checkpoint to Modal Volume…"
echo "  (model_objaverse.ckpt ~500 MB from HuggingFace)"
echo ""
modal run modal_app.py::download_weights
echo ""
echo "  ✓ Checkpoint stored in Modal Volume 'partfield-data'"
echo ""

# ── 5. Deploy ─────────────────────────────────────────────────────────────
echo "▶ Deploying PartField Studio to Modal…"
echo ""
modal deploy modal_app.py
echo ""

echo "═══════════════════════════════════════════════"
echo "  ✓ Deployment complete!"
echo ""
echo "  Your app URL will be shown above as:"
echo "  https://YOUR-USERNAME--partfield-studio-web.modal.run"
echo ""
echo "  To get the URL again:  modal app list"
echo "  To view logs:          modal app logs partfield-studio"
echo "  For dev/hot-reload:    modal serve modal_app.py"
echo "═══════════════════════════════════════════════"
