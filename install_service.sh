#!/bin/bash
# install_service.sh
# Installs the Ornimetrics detection system as a systemd service for auto-start on boot

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

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
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

echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════╗"
echo "║   Ornimetrics Detection Service Installation          ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Get username (the user who ran sudo)
ACTUAL_USER="${SUDO_USER:-$USER}"
if [ "$ACTUAL_USER" = "root" ]; then
    log_warn "Running as root. Assuming user 'pi'"
    ACTUAL_USER="pi"
fi

# Get installation directory
INSTALL_DIR="/home/$ACTUAL_USER/Ornimetrics-AI"

log_info "Installation settings:"
echo "  User: $ACTUAL_USER"
echo "  Installation directory: $INSTALL_DIR"
echo ""

# Check if directory exists
if [ ! -d "$INSTALL_DIR" ]; then
    log_error "Directory $INSTALL_DIR does not exist!"
    echo ""
    echo "Please ensure the Ornimetrics-AI repository is cloned to:"
    echo "  $INSTALL_DIR"
    exit 1
fi

# Check if service file exists
SERVICE_FILE="ornimetrics-detection.service"
if [ ! -f "$SERVICE_FILE" ]; then
    log_error "Service file $SERVICE_FILE not found in current directory"
    exit 1
fi

# Update service file with correct paths and user
log_info "Configuring service file..."
TEMP_SERVICE="/tmp/ornimetrics-detection.service"
sed -e "s|User=pi|User=$ACTUAL_USER|g" \
    -e "s|Group=pi|Group=$ACTUAL_USER|g" \
    -e "s|/home/pi/Ornimetrics-AI|$INSTALL_DIR|g" \
    "$SERVICE_FILE" > "$TEMP_SERVICE"

# Copy service file to systemd directory
log_info "Installing service file..."
cp "$TEMP_SERVICE" /etc/systemd/system/ornimetrics-detection.service
chmod 644 /etc/systemd/system/ornimetrics-detection.service
rm "$TEMP_SERVICE"
log_success "Service file installed"

# Reload systemd
log_info "Reloading systemd daemon..."
systemctl daemon-reload
log_success "Systemd reloaded"

# Make scripts executable
log_info "Making scripts executable..."
chmod +x "$INSTALL_DIR/start_detection_system.sh"
chmod +x "$INSTALL_DIR/web_detection_server.py"
chmod +x "$INSTALL_DIR/detect_3d_individual.py"
log_success "Scripts are executable"

# Ask if user wants to enable auto-start
echo ""
read -p "Enable auto-start on boot? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Enabling service..."
    systemctl enable ornimetrics-detection.service
    log_success "Service enabled - will start automatically on boot"
else
    log_info "Service installed but not enabled"
    log_info "To enable later, run: sudo systemctl enable ornimetrics-detection"
fi

# Ask if user wants to start now
echo ""
read -p "Start the service now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Starting service..."
    systemctl start ornimetrics-detection.service
    sleep 2

    # Check status
    if systemctl is-active --quiet ornimetrics-detection.service; then
        log_success "Service started successfully!"
        echo ""
        log_info "Dashboard should be available at:"
        echo -e "  ${GREEN}http://$(hostname -I | awk '{print $1}'):5000/${NC}"
        echo ""
        log_info "Useful commands:"
        echo "  sudo systemctl status ornimetrics-detection   # Check status"
        echo "  sudo systemctl stop ornimetrics-detection     # Stop service"
        echo "  sudo systemctl restart ornimetrics-detection  # Restart service"
        echo "  sudo journalctl -u ornimetrics-detection -f   # View logs"
    else
        log_error "Service failed to start. Check logs with:"
        echo "  sudo journalctl -u ornimetrics-detection -n 50"
    fi
else
    log_info "Service not started. To start later, run:"
    echo "  sudo systemctl start ornimetrics-detection"
fi

echo ""
log_success "Installation complete!"
echo ""
log_info "Configuration file: $INSTALL_DIR/config_3d_detection.json"
log_info "Service file: /etc/systemd/system/ornimetrics-detection.service"
echo ""

# Additional setup recommendations
log_info "Additional recommendations:"
echo "  1. Calibrate camera intrinsics in config_3d_detection.json"
echo "  2. Update model path in config if needed"
echo "  3. Set up firewall rules if needed:"
echo "     sudo ufw allow 5000/tcp"
echo "  4. For production, consider changing host to 127.0.0.1 in config"
echo ""
