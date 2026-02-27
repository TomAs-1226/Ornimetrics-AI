#!/bin/bash
# setup.sh — One-command setup for Ornimetrics on Raspberry Pi
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# What it does:
#   1. Checks Python 3.8+
#   2. Installs pip dependencies
#   3. Creates data directories
#   4. Detects hardware (Hailo, cameras)
#   5. Prints how to run

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; B='\033[0;34m'; N='\033[0m'

echo -e "${G}Ornimetrics Setup${N}"
echo "================================"

# 1. Python check
echo -e "\n${B}[1/4]${N} Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo -e "${R}Python 3 not found. Install with: sudo apt install python3 python3-pip${N}"
    exit 1
fi
PV=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "  ${G}OK${N} Python $PV"

# 2. Install dependencies
echo -e "\n${B}[2/4]${N} Installing dependencies..."
python3 -m pip install --quiet --upgrade pip 2>/dev/null || true
python3 -m pip install --quiet -r requirements.txt 2>&1 | tail -1 || {
    echo -e "${Y}Some packages failed — trying individually...${N}"
    while read -r pkg; do
        pkg=$(echo "$pkg" | sed 's/#.*//' | xargs)
        [ -z "$pkg" ] && continue
        python3 -m pip install --quiet "$pkg" 2>/dev/null || echo -e "  ${Y}skip${N} $pkg"
    done < requirements.txt
}
echo -e "  ${G}OK${N} Dependencies installed"

# 3. Create directories
echo -e "\n${B}[3/4]${N} Creating directories..."
mkdir -p data logs models backups
echo -e "  ${G}OK${N} data/ logs/ models/ backups/"

# 4. Hardware detection
echo -e "\n${B}[4/4]${N} Detecting hardware..."

# Hailo
if python3 -c "import hailo_platform" 2>/dev/null; then
    echo -e "  ${G}OK${N} Hailo-8 AI Hat detected"
else
    echo -e "  ${Y}--${N} Hailo not found (will use PyTorch CPU)"
fi

# Camera
if ls /dev/video* &>/dev/null; then
    echo -e "  ${G}OK${N} Camera(s) found: $(ls /dev/video* 2>/dev/null | wc -l) device(s)"
else
    echo -e "  ${Y}--${N} No video devices found"
fi

# Model
if [ -f "weights.pt" ] || [ -f "best.pt" ]; then
    echo -e "  ${G}OK${N} YOLO model found"
else
    echo -e "  ${Y}--${N} No model file found (place weights.pt in project root)"
fi

# Done
echo -e "\n${G}================================${N}"
echo -e "${G}Setup complete!${N}\n"
echo "Quick start:"
echo "  python3 run.py                  # headless detection"
echo "  python3 run.py --web            # web dashboard on :5000"
echo "  python3 run.py --preview        # local GUI preview"
echo "  python3 run.py --no-3d          # skip depth camera"
echo "  python3 run.py --quiet          # minimal logs"
echo ""
echo "Config: config_3d_detection.json"
echo "Trap settings: trap_settings_full.json"
echo ""
