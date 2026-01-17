#!/bin/bash
# install_service_auto.sh
# Non-interactive installation of Ornimetrics OS service for auto-start on boot
# This script is called automatically during OOBE setup

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log_error "This script must be run as root (use sudo)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log_info "Installing Ornimetrics OS service..."

# Get username (the user who ran sudo)
ACTUAL_USER="${SUDO_USER:-$USER}"
if [ "$ACTUAL_USER" = "root" ]; then
    # Try to detect from the directory ownership
    ACTUAL_USER=$(stat -c '%U' "$SCRIPT_DIR" 2>/dev/null || echo "pi")
fi

# Get installation directory
INSTALL_DIR="$SCRIPT_DIR"

log_info "Configuration:"
echo "  User: $ACTUAL_USER"
echo "  Directory: $INSTALL_DIR"

# Check if service file exists
SERVICE_FILE="ornimetrics-detection.service"
if [ ! -f "$SERVICE_FILE" ]; then
    log_error "Service file $SERVICE_FILE not found"
    exit 1
fi

# Update service file with correct paths and user
log_info "Configuring service..."
TEMP_SERVICE="/tmp/ornimetrics-detection.service"
sed -e "s|User=pi|User=$ACTUAL_USER|g" \
    -e "s|Group=pi|Group=$ACTUAL_USER|g" \
    -e "s|/home/pi/Ornimetrics-AI|$INSTALL_DIR|g" \
    "$SERVICE_FILE" > "$TEMP_SERVICE"

# Copy service file to systemd directory
cp "$TEMP_SERVICE" /etc/systemd/system/ornimetrics-detection.service
chmod 644 /etc/systemd/system/ornimetrics-detection.service
rm "$TEMP_SERVICE"

# Reload systemd
systemctl daemon-reload

# Make scripts executable
chmod +x "$INSTALL_DIR/start_detection_system.sh" 2>/dev/null || true
chmod +x "$INSTALL_DIR/web_detection_server.py" 2>/dev/null || true
chmod +x "$INSTALL_DIR/detect_3d_individual.py" 2>/dev/null || true
chmod +x "$INSTALL_DIR/oobe_ornimetrics_os.py" 2>/dev/null || true

# Enable service for auto-start on boot
log_info "Enabling auto-start on boot..."
systemctl enable ornimetrics-detection.service

# Start the service now
log_info "Starting service..."
systemctl start ornimetrics-detection.service
sleep 2

# Check status
if systemctl is-active --quiet ornimetrics-detection.service; then
    log_success "Ornimetrics OS service installed and running!"
    log_success "Will automatically start on every boot"
    echo ""
    log_info "Useful commands:"
    echo "  sudo systemctl status ornimetrics-detection   # Check status"
    echo "  sudo systemctl stop ornimetrics-detection     # Stop service"
    echo "  sudo systemctl restart ornimetrics-detection  # Restart service"
    echo "  sudo journalctl -u ornimetrics-detection -f   # View logs"
    exit 0
else
    log_error "Service failed to start. Check logs:"
    echo "  sudo journalctl -u ornimetrics-detection -n 50"
    exit 1
fi
