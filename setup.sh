#!/bin/bash
# setup.sh — One-command setup for Ornimetrics on Raspberry Pi 5
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# What it does:
#   1. Checks system requirements (Pi 5, Python 3.9+)
#   2. Installs Hailo AI Hat+ drivers (hailo-all) if not present
#   3. Creates a Python venv with --system-site-packages (required for hailo_platform)
#   4. Installs pip dependencies inside the venv
#   5. Creates data directories and models/ folder
#   6. Detects hardware (Hailo, cameras, servo)
#   7. Prints how to run

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; B='\033[0;34m'; N='\033[0m'
VENV_DIR="$DIR/venv"

echo -e "${G}Ornimetrics Setup${N}"
echo "================================"

# ── 1. System check ──────────────────────────────────────────────────
echo -e "\n${B}[1/6]${N} Checking system..."

if ! command -v python3 &>/dev/null; then
    echo -e "${R}Python 3 not found. Install: sudo apt install python3 python3-pip python3-venv${N}"
    exit 1
fi
PV=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PV_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PV_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PV_MAJOR" -lt 3 ] || ([ "$PV_MAJOR" -eq 3 ] && [ "$PV_MINOR" -lt 9 ]); then
    echo -e "${R}Python 3.9+ required, found $PV${N}"
    exit 1
fi
echo -e "  ${G}OK${N} Python $PV"

# Check if running on Raspberry Pi
IS_PI=false
if grep -qi "raspberry" /proc/cpuinfo 2>/dev/null || grep -qi "raspberry" /proc/device-tree/model 2>/dev/null; then
    IS_PI=true
    PI_MODEL=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0' || echo "unknown")
    echo -e "  ${G}OK${N} $PI_MODEL"
else
    echo -e "  ${Y}--${N} Not a Raspberry Pi (some features may not work)"
fi

# ── 2. Hailo AI Hat+ driver install ─────────────────────────────────
echo -e "\n${B}[2/6]${N} Checking Hailo AI Hat+ drivers..."

HAILO_INSTALLED=false
if python3 -c "import hailo_platform" 2>/dev/null; then
    HV=$(python3 -c "
try:
    from hailo_platform import __version__; print(__version__)
except: print('installed')
" 2>/dev/null)
    echo -e "  ${G}OK${N} hailo_platform $HV"
    HAILO_INSTALLED=true
elif dpkg -l hailo-all 2>/dev/null | grep -q "^ii"; then
    echo -e "  ${G}OK${N} hailo-all package installed (reboot may be needed)"
    HAILO_INSTALLED=true
else
    echo -e "  ${Y}--${N} Hailo runtime not found"
    if [ "$IS_PI" = true ]; then
        echo ""
        echo -e "  The Hailo AI Hat+ needs drivers to work."
        echo -e "  This will install: hailo-dkms, hailort, python3-hailort, hailo-tappas-core"
        echo ""
        read -p "  Install Hailo drivers now? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "  Installing Hailo drivers (this may take a few minutes)..."
            sudo apt update -qq
            # Ensure dkms is available for kernel module
            sudo apt install -y -qq dkms 2>/dev/null || true
            if sudo apt install -y hailo-all 2>&1 | tail -5; then
                echo -e "  ${G}OK${N} Hailo drivers installed"
                echo -e "  ${Y}NOTE:${N} You must reboot for the driver to load: ${B}sudo reboot${N}"
                HAILO_INSTALLED=true
            else
                echo -e "  ${R}Failed to install hailo-all${N}"
                echo -e "  Make sure you are on Raspberry Pi OS (Bookworm or newer)"
                echo -e "  Try: sudo apt update && sudo apt full-upgrade && sudo apt install hailo-all"
            fi
        else
            echo -e "  Skipping — will use CPU inference (slower)"
        fi
    else
        echo -e "  Hailo only works on Raspberry Pi 5 — skipping driver install"
    fi
fi

# Check for /dev/hailo0 device node
if [ -e /dev/hailo0 ]; then
    echo -e "  ${G}OK${N} /dev/hailo0 present (Hailo device detected)"
else
    if [ "$HAILO_INSTALLED" = true ]; then
        echo -e "  ${Y}--${N} /dev/hailo0 not found — reboot needed, or Hat+ not connected"
    fi
fi

# ── 3. Python virtual environment ───────────────────────────────────
echo -e "\n${B}[3/6]${N} Setting up Python environment..."

# Ensure python3-venv is available
if ! python3 -m venv --help &>/dev/null; then
    echo -e "  Installing python3-venv..."
    sudo apt install -y -qq python3-venv 2>/dev/null || true
fi

if [ ! -d "$VENV_DIR" ]; then
    echo -e "  Creating venv with --system-site-packages (required for hailo_platform)..."
    python3 -m venv "$VENV_DIR" --system-site-packages
    echo -e "  ${G}OK${N} venv created at $VENV_DIR"
else
    echo -e "  ${G}OK${N} venv already exists at $VENV_DIR"
fi

# Activate venv for the rest of setup
source "$VENV_DIR/bin/activate"
echo -e "  ${G}OK${N} venv activated (Python: $(which python3))"

# ── 4. Install pip dependencies ─────────────────────────────────────
echo -e "\n${B}[4/6]${N} Installing Python packages..."

python3 -m pip install --quiet --upgrade pip 2>/dev/null || true

# Install core packages one by one so failures don't block everything
CORE_PKGS="numpy opencv-python-headless flask flask-cors sqlite-utils pytest"
for pkg in $CORE_PKGS; do
    if python3 -c "import $(echo $pkg | sed 's/-/_/g' | sed 's/_headless//')" 2>/dev/null; then
        echo -e "  ${G}OK${N} $pkg (already installed)"
    else
        echo -e "  Installing $pkg..."
        python3 -m pip install --quiet "$pkg" 2>/dev/null && \
            echo -e "  ${G}OK${N} $pkg" || \
            echo -e "  ${Y}skip${N} $pkg (install failed)"
    fi
done

# Optional heavy packages — skip gracefully if they fail (ARM builds can be slow)
echo -e "  Installing optional packages (may take a while on ARM)..."
for pkg in ultralytics torch torchvision; do
    if python3 -c "import $pkg" 2>/dev/null; then
        echo -e "  ${G}OK${N} $pkg (already installed)"
    else
        echo -e "  Installing $pkg..."
        python3 -m pip install --quiet "$pkg" 2>&1 | tail -1 || \
            echo -e "  ${Y}skip${N} $pkg (optional — Hailo inference does not need PyTorch)"
    fi
done

# ── 5. Create directories ───────────────────────────────────────────
echo -e "\n${B}[5/6]${N} Creating directories..."
mkdir -p data logs models backups
echo -e "  ${G}OK${N} data/ logs/ models/ backups/"

# ── 6. Hardware detection ───────────────────────────────────────────
echo -e "\n${B}[6/6]${N} Detecting hardware..."

# Hailo device
if [ -e /dev/hailo0 ]; then
    CHIP="unknown"
    if command -v hailortcli &>/dev/null; then
        CHIP=$(hailortcli fw-control identify 2>/dev/null | grep "Device Architecture" | awk '{print $NF}' || echo "unknown")
    fi
    echo -e "  ${G}OK${N} Hailo accelerator detected ($CHIP)"
elif [ "$HAILO_INSTALLED" = true ]; then
    echo -e "  ${Y}--${N} Hailo drivers installed but device not found (reboot or check Hat+ connection)"
else
    echo -e "  ${Y}--${N} Hailo not available (will use CPU)"
fi

# Camera
CAM_COUNT=$(ls /dev/video* 2>/dev/null | wc -l)
if [ "$CAM_COUNT" -gt 0 ]; then
    echo -e "  ${G}OK${N} Camera(s) found: $CAM_COUNT device(s)"
else
    echo -e "  ${Y}--${N} No video devices found"
fi

# Model files
HEF_FOUND=false
PT_FOUND=false
for f in models/model.hef models/best.hef model.hef best.hef; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo -e "  ${G}OK${N} HEF model: $f ($SIZE)"
        HEF_FOUND=true
        break
    fi
done
for f in models/model.pt models/best.pt models/weights.pt model.pt best.pt weights.pt; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo -e "  ${G}OK${N} PT model: $f ($SIZE)"
        PT_FOUND=true
        break
    fi
done
if [ "$HEF_FOUND" = false ] && [ "$PT_FOUND" = false ]; then
    echo -e "  ${Y}--${N} No model files found"
    echo -e "       Place your model in: ${B}models/model.hef${N} (Hailo) or ${B}models/model.pt${N} (CPU)"
fi

# Servo
if python3 -c "from adafruit_servokit import ServoKit" 2>/dev/null; then
    echo -e "  ${G}OK${N} Servo libraries available"
else
    echo -e "  ${Y}--${N} adafruit_servokit not installed (trap door will be disabled)"
    echo -e "       Install: pip install adafruit-circuitpython-servokit"
fi

# ── Done ─────────────────────────────────────────────────────────────
echo -e "\n${G}================================${N}"
echo -e "${G}Setup complete!${N}\n"
echo "To activate the environment:"
echo -e "  ${B}source venv/bin/activate${N}"
echo ""
echo "Quick start:"
echo -e "  ${B}python3 run.py --web${N}            # web dashboard on :5000"
echo -e "  ${B}python3 run.py${N}                  # headless detection"
echo -e "  ${B}python3 run.py --preview${N}        # local GUI preview"
echo -e "  ${B}python3 run.py --no-3d${N}          # skip depth processing"
echo -e "  ${B}python3 run.py --quiet${N}          # minimal logs"
echo ""
echo "Config: config_3d_detection.json"
echo "Models: models/model.hef (Hailo) or models/model.pt (CPU fallback)"
echo ""
if [ "$HAILO_INSTALLED" = true ] && [ ! -e /dev/hailo0 ]; then
    echo -e "${Y}IMPORTANT: Reboot required for Hailo driver to load:${N}"
    echo -e "  ${B}sudo reboot${N}"
    echo ""
fi
