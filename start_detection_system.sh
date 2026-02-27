#!/bin/bash
# start_detection_system.sh
# Unified startup script for Ornimetrics bird detection system
# This script auto-detects hardware and starts the web server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════╗"
echo "║              ORNIMETRICS OS v1.0.0                    ║"
echo "║     Smart Bird Feeder with Individual Recognition    ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check for OOBE (Out Of Box Experience) completion
if [ ! -f ".oobe_completed" ]; then
    log_info "First time setup detected - Starting Ornimetrics OS OOBE..."
    echo ""

    # Use Ornimetrics OS OOBE with Bluetooth setup
    if [ -f "oobe_ornimetrics_os.py" ]; then
        python3 oobe_ornimetrics_os.py || {
            log_error "OOBE setup failed. Please run manually: python3 oobe_ornimetrics_os.py"
            exit 1
        }
    elif [ -f "oobe_setup.py" ]; then
        log_warn "Using legacy OOBE setup (Ornimetrics OS OOBE not found)..."
        python3 oobe_setup.py || {
            log_error "OOBE setup failed. Please run manually: python3 oobe_setup.py"
            exit 1
        }
    else
        log_warn "OOBE setup script not found, continuing anyway..."
    fi
    echo ""
fi

# Check Python
log_info "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
log_success "Python $PYTHON_VERSION found"

# Check required Python packages
log_info "Checking required packages..."
MISSING_PACKAGES=()

check_package() {
    if ! python3 -c "import $1" &> /dev/null; then
        MISSING_PACKAGES+=("$2")
        return 1
    fi
    return 0
}

check_package "cv2" "opencv-python"
check_package "numpy" "numpy"
check_package "flask" "flask"
check_package "ultralytics" "ultralytics"

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    log_warn "Missing packages: ${MISSING_PACKAGES[*]}"
    log_info "Attempting to install missing packages..."
    pip3 install "${MISSING_PACKAGES[@]}" || {
        log_error "Failed to install packages. Please run: pip3 install ${MISSING_PACKAGES[*]}"
        exit 1
    }
    log_success "Packages installed"
else
    log_success "All required packages installed"
fi

# Check hardware
log_info "Detecting hardware..."

# Check for CS20 depth camera
if ls /dev/video* &> /dev/null; then
    VIDEO_DEVICES=$(ls /dev/video* | wc -l)
    log_success "Found $VIDEO_DEVICES video device(s)"

    # Try to detect depth camera specifically
    DEPTH_CAM_FOUND=false
    for dev in /dev/video*; do
        # This is a simple check - actual detection happens in Python
        if v4l2-ctl --device="$dev" --all 2>/dev/null | grep -iq "depth\|tof\|cs20"; then
            log_success "Depth camera detected at $dev"
            DEPTH_CAM_FOUND=true
            break
        fi
    done

    if [ "$DEPTH_CAM_FOUND" = false ]; then
        log_warn "Depth camera not detected (will use optics-only mode)"
    fi
else
    log_warn "No video devices found at /dev/video*"
fi

# Check for Hailo
if [ -d "/usr/lib/hailo" ] || python3 -c "import hailo_platform" &> /dev/null 2>&1; then
    log_success "Hailo runtime detected"
else
    log_warn "Hailo runtime not detected (will use PyTorch CPU)"
fi

# Check for models
log_info "Checking for YOLO models..."
MODEL_FOUND=false

# Check common model locations
MODEL_PATHS=(
    "/home/pi/Desktop/FinalPrototype 2/FinalPrototype/GoodModel/weights(2).pt"
    "/home/pi/Desktop/FinalPrototype/weights.pt"
    "best.pt"
    "weights.pt"
    "yolov8n.pt"
)

for model_path in "${MODEL_PATHS[@]}"; do
    if [ -f "$model_path" ]; then
        log_success "Model found: $model_path"
        MODEL_FOUND=true
        break
    fi
done

if [ "$MODEL_FOUND" = false ]; then
    log_warn "No YOLO model found in common locations"
    log_info "Make sure to specify --model path or update config_3d_detection.json"
fi

# Check configuration
if [ -f "config_3d_detection.json" ]; then
    log_success "Configuration file found"
else
    log_warn "config_3d_detection.json not found (will use defaults)"
fi

# Check if already running
if pgrep -f "web_detection_server.py" > /dev/null; then
    log_warn "Detection server already running!"
    log_info "To restart, first stop the existing process:"
    echo "  sudo systemctl stop ornimetrics-detection"
    echo "  OR"
    echo "  pkill -f web_detection_server.py"
    exit 1
fi

# Parse arguments
PORT=5000
HOST="0.0.0.0"
CONFIG="config_3d_detection.json"

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--port PORT] [--host HOST] [--config CONFIG]"
            exit 1
            ;;
    esac
done

# Start the server
echo ""
log_success "Starting Ornimetrics OS..."
echo ""
log_info "Dashboard will be available at:"
echo -e "  ${GREEN}http://$HOST:$PORT/${NC}"
echo ""
log_info "Streaming endpoints:"
echo -e "  MJPEG: ${GREEN}http://$HOST:$PORT/video_feed${NC}"
echo -e "  API:   ${GREEN}http://$HOST:$PORT/api/status${NC}"
echo ""
log_info "Press Ctrl+C to stop"
echo ""

# Set environment for headless operation
export QT_QPA_PLATFORM=offscreen
export DISPLAY=:0

# Start the server (use unified run.py entry point)
python3 run.py \
    --config "$CONFIG" \
    --web \
    --port "$PORT"
