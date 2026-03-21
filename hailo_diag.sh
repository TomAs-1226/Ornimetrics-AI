#!/bin/bash
# hailo_diag.sh — Diagnose why Hailo AI Hat is not being detected
#
# Run this on your Raspberry Pi:
#   chmod +x hailo_diag.sh && ./hailo_diag.sh

G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; B='\033[0;34m'; N='\033[0m'

ok()   { echo -e "  ${G}[OK]${N}   $1"; }
warn() { echo -e "  ${Y}[WARN]${N} $1"; }
fail() { echo -e "  ${R}[FAIL]${N} $1"; }
info() { echo -e "  ${B}[INFO]${N} $1"; }

echo -e "${B}Ornimetrics — Hailo Diagnostics${N}"
echo "========================================"
ISSUES=0

# ── 1. Are we on a Pi? ─────────────────────────────────────────────────
echo -e "\n${B}[1] Raspberry Pi check${N}"
if grep -qi "raspberry" /proc/cpuinfo 2>/dev/null || grep -qi "raspberry" /proc/device-tree/model 2>/dev/null; then
    MODEL=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0' || echo "unknown")
    ok "$MODEL"
    if echo "$MODEL" | grep -qi "pi 5"; then
        ok "Pi 5 confirmed — Hailo AI Hat+ is supported"
    else
        warn "Hailo AI Hat+ is designed for Pi 5 — other models may have issues"
        ISSUES=$((ISSUES+1))
    fi
else
    warn "Not on a Raspberry Pi — Hailo will not work"
    ISSUES=$((ISSUES+1))
fi

# ── 2. PCIe enabled in config.txt? ────────────────────────────────────
echo -e "\n${B}[2] PCIe / boot config check${N}"
CONFIG_FILE=""
for f in /boot/firmware/config.txt /boot/config.txt; do
    [ -f "$f" ] && CONFIG_FILE="$f" && break
done

if [ -z "$CONFIG_FILE" ]; then
    fail "Cannot find config.txt — checked /boot/firmware/config.txt and /boot/config.txt"
    ISSUES=$((ISSUES+1))
else
    info "Config file: $CONFIG_FILE"

    # Check PCIe dtparam
    if grep -qE "^\s*dtparam=pciex1" "$CONFIG_FILE" 2>/dev/null; then
        ok "PCIe enabled (dtparam=pciex1 found)"
    else
        fail "PCIe NOT enabled in $CONFIG_FILE"
        echo -e "       ${Y}FIX:${N} Add the following lines to $CONFIG_FILE:"
        echo -e "           ${B}dtparam=pciex1${N}"
        echo -e "       Then reboot: ${B}sudo reboot${N}"
        ISSUES=$((ISSUES+1))
    fi

    # Check for gen 3 override (can cause instability with Hailo)
    if grep -qE "^\s*dtparam=pciex1_gen=3" "$CONFIG_FILE" 2>/dev/null; then
        warn "PCIe Gen 3 override found — Hailo recommends Gen 2"
        echo -e "       ${Y}FIX:${N} Change to: ${B}dtparam=pciex1_gen=2${N} in $CONFIG_FILE"
        ISSUES=$((ISSUES+1))
    fi
fi

# ── 3. hailo-all package installed? ───────────────────────────────────
echo -e "\n${B}[3] Hailo apt package check${N}"
if dpkg -l hailo-all 2>/dev/null | grep -q "^ii"; then
    PKG_VER=$(dpkg -l hailo-all 2>/dev/null | grep "^ii" | awk '{print $3}')
    ok "hailo-all installed ($PKG_VER)"
else
    fail "hailo-all NOT installed"
    echo -e "       ${Y}FIX:${N} Run:"
    echo -e "           ${B}sudo apt update && sudo apt install hailo-all${N}"
    echo -e "       Then reboot: ${B}sudo reboot${N}"
    ISSUES=$((ISSUES+1))
fi

# Check HailoRT cli tool
if command -v hailortcli &>/dev/null; then
    ok "hailortcli found: $(which hailortcli)"
else
    warn "hailortcli not in PATH (may still work)"
fi

# ── 4. Kernel module loaded? ──────────────────────────────────────────
echo -e "\n${B}[4] Kernel driver check${N}"
if lsmod 2>/dev/null | grep -q hailo; then
    HAILO_MOD=$(lsmod | grep hailo | awk '{print $1}')
    ok "Hailo kernel module loaded: $HAILO_MOD"
else
    fail "Hailo kernel module NOT loaded"
    echo -e "       ${Y}FIX:${N} Try loading it manually: ${B}sudo modprobe hailo_pci${N}"
    echo -e "       If that fails, run: ${B}sudo apt reinstall hailo-all && sudo reboot${N}"
    ISSUES=$((ISSUES+1))
fi

# ── 5. /dev/hailo0 device node? ───────────────────────────────────────
echo -e "\n${B}[5] Device node check${N}"
if [ -e /dev/hailo0 ]; then
    ok "/dev/hailo0 exists"
    ls -la /dev/hailo0 | awk '{print "       perms: "$1"  owner: "$3":"$4}'
else
    fail "/dev/hailo0 NOT found"
    echo -e "       This means the PCIe device is not being recognised."
    echo -e "       ${Y}Steps to try:${N}"
    echo -e "         1. Enable PCIe in /boot/firmware/config.txt (see step 2 above)"
    echo -e "         2. ${B}sudo reboot${N}"
    echo -e "         3. Check Hat+ is firmly seated in the M.2 slot"
    echo -e "         4. ${B}lspci | grep -i hailo${N} — if nothing, it's a hardware/PCIe issue"
    ISSUES=$((ISSUES+1))
fi

# ── 6. User permissions ────────────────────────────────────────────────
echo -e "\n${B}[6] User permissions check${N}"
CURRENT_USER=$(whoami)
if id -nG "$CURRENT_USER" 2>/dev/null | grep -qw "hailo"; then
    ok "User '$CURRENT_USER' is in the 'hailo' group"
elif [ -e /dev/hailo0 ]; then
    DEV_GROUP=$(stat -c "%G" /dev/hailo0 2>/dev/null || echo "unknown")
    DEV_PERMS=$(stat -c "%a" /dev/hailo0 2>/dev/null || echo "unknown")
    if [ "$DEV_GROUP" = "hailo" ]; then
        fail "User '$CURRENT_USER' is NOT in the 'hailo' group (device group: $DEV_GROUP)"
        echo -e "       ${Y}FIX:${N} ${B}sudo usermod -aG hailo $CURRENT_USER${N}"
        echo -e "       Then log out and back in (or: ${B}newgrp hailo${N})"
        ISSUES=$((ISSUES+1))
    elif [ "$DEV_PERMS" = "666" ] || [ "$DEV_PERMS" = "660" ]; then
        ok "Device permissions look ok ($DEV_PERMS)"
    else
        warn "hailo group not found — permissions: $DEV_PERMS (owner/group: $DEV_GROUP)"
    fi
else
    info "Skipping group check (/dev/hailo0 not present)"
fi

# ── 7. PCIe bus detection ──────────────────────────────────────────────
echo -e "\n${B}[7] PCIe hardware scan${N}"
if command -v lspci &>/dev/null; then
    HAILO_PCI=$(lspci -d "1e60:" 2>/dev/null)
    if [ -n "$HAILO_PCI" ]; then
        ok "Hailo device found on PCIe bus:"
        echo "       $HAILO_PCI"
    else
        ALL_PCI=$(lspci 2>/dev/null | head -20)
        fail "No Hailo device found on PCIe (vendor 1e60)"
        echo -e "       ${Y}Devices visible on PCIe:${N}"
        echo "$ALL_PCI" | while IFS= read -r line; do echo "         $line"; done
        echo -e "       If no devices at all, PCIe is likely disabled in config.txt"
        ISSUES=$((ISSUES+1))
    fi
else
    warn "lspci not installed — install with: ${B}sudo apt install pciutils${N}"
    ISSUES=$((ISSUES+1))
fi

# ── 8. hailortcli firmware check ──────────────────────────────────────
echo -e "\n${B}[8] Hailo firmware check${N}"
if command -v hailortcli &>/dev/null && [ -e /dev/hailo0 ]; then
    FW_INFO=$(hailortcli fw-control identify 2>&1)
    if echo "$FW_INFO" | grep -qi "error\|fail\|denied"; then
        fail "hailortcli fw-control identify failed:"
        echo "$FW_INFO" | head -5 | while IFS= read -r line; do echo "       $line"; done
        ISSUES=$((ISSUES+1))
    else
        ok "Hailo firmware responded:"
        echo "$FW_INFO" | grep -E "Device|Architecture|Firmware" | while IFS= read -r line; do echo "       $line"; done
    fi
elif [ -e /dev/hailo0 ]; then
    warn "hailortcli not found — skipping firmware check"
else
    info "Skipping firmware check (/dev/hailo0 not present)"
fi

# ── 9. Python hailo_platform importable? ──────────────────────────────
echo -e "\n${B}[9] Python hailo_platform check${N}"

# Check system Python
SYS_PY=$(which python3)
info "System Python: $SYS_PY"
if $SYS_PY -c "import hailo_platform; print('version:', getattr(hailo_platform, '__version__', 'unknown'))" 2>/dev/null; then
    ok "hailo_platform importable from system Python"
else
    IMPORT_ERR=$($SYS_PY -c "import hailo_platform" 2>&1)
    fail "hailo_platform NOT importable from system Python"
    echo "       Error: $IMPORT_ERR"
    ISSUES=$((ISSUES+1))
fi

# Check venv if it exists
VENV_DIR="$(dirname "${BASH_SOURCE[0]}")/venv"
if [ -d "$VENV_DIR" ]; then
    VENV_PY="$VENV_DIR/bin/python3"
    info "Venv Python: $VENV_PY"
    if $VENV_PY -c "import hailo_platform" 2>/dev/null; then
        ok "hailo_platform importable from venv"
    else
        IMPORT_ERR=$($VENV_PY -c "import hailo_platform" 2>&1)
        fail "hailo_platform NOT importable from venv"
        echo "       Error: $IMPORT_ERR"
        echo -e "       ${Y}FIX:${N} Venv must be created with ${B}--system-site-packages${N}:"
        echo -e "           ${B}rm -rf venv && python3 -m venv venv --system-site-packages${N}"
        echo -e "       (hailo_platform is installed by apt, not pip)"
        ISSUES=$((ISSUES+1))
    fi
else
    info "No venv found at $VENV_DIR — run ./setup.sh first"
fi

# ── Summary ────────────────────────────────────────────────────────────
echo ""
echo "========================================"
if [ "$ISSUES" -eq 0 ]; then
    echo -e "${G}All checks passed — Hailo should be working!${N}"
    echo ""
    echo "Test with:"
    echo -e "  ${B}python3 -c \"from src.hailo_detector import hailo_device_available; print(hailo_device_available())\"${N}"
else
    echo -e "${R}Found $ISSUES issue(s) above.${N}"
    echo ""
    echo -e "Most common fix for a freshly-installed Pi:"
    echo -e "  ${B}sudo apt update && sudo apt install hailo-all pciutils${N}"
    echo -e "  Add ${B}dtparam=pciex1${N} to /boot/firmware/config.txt"
    echo -e "  ${B}sudo reboot${N}"
fi
echo ""
