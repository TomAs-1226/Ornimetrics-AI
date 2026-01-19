# Ornimetrics OS - Auto-Start Guide

Ornimetrics OS is designed to start automatically on power-on with **zero manual intervention**. No SSH login, no passwords, no manual commands needed.

---

## How It Works

### 1. Power On → Automatic Boot

When you plug in power to your Raspberry Pi:

```
1. Raspberry Pi boots up
2. Linux starts
3. Systemd launches Ornimetrics OS service automatically
4. Everything is ready in 30-60 seconds
```

**No login required. No SSH needed. No keyboard/monitor needed.**

---

## What Happens on First Boot?

### First Time Setup (OOBE)

The very first time you power on your device:

```
1. Power on device
2. System detects first boot (no .oobe_completed marker)
3. OOBE (Out-Of-Box Experience) starts automatically
4. Bluetooth service starts for mobile app pairing
5. Wait for mobile app to:
   - Pair with device
   - Link to your account
   - Configure WiFi
6. System installs auto-start service
7. System marks setup complete
8. Ornimetrics OS starts automatically
9. On every subsequent boot, system starts instantly
```

**After first boot: Device starts automatically forever.**

---

## Auto-Start Configuration

### Systemd Service

Ornimetrics OS uses systemd (Linux's standard service manager) for automatic startup.

**Service File:** `/etc/systemd/system/ornimetrics-detection.service`

**Key Settings:**
```ini
[Unit]
Description=Ornimetrics OS - Smart Bird Feeder
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /home/pi/Ornimetrics-AI/web_detection_server.py
Restart=always  # <-- Automatically restarts if it crashes
RestartSec=10

[Install]
WantedBy=multi-user.target  # <-- Starts on boot
```

---

## Installation Methods

### Automatic (During OOBE)

OOBE automatically installs the service at the end of setup:

```bash
# Happens automatically during OOBE
sudo ./install_service_auto.sh
```

**Result:** Service enabled and started, will auto-start on every boot.

---

### Manual Installation

If you need to install manually (skip OOBE, or reinstall):

**Interactive (with prompts):**
```bash
sudo ./install_service.sh
```

**Non-interactive (automatic yes):**
```bash
sudo ./install_service_auto.sh
```

**Result:** Service enabled and started, will auto-start on every boot.

---

## Service Management Commands

### Check Status

```bash
sudo systemctl status ornimetrics-detection
```

**Shows:**
- Running or stopped
- Uptime
- Recent log messages

---

### View Live Logs

```bash
sudo journalctl -u ornimetrics-detection -f
```

**Shows:**
- Real-time log output
- Detection events
- Errors/warnings

---

### Stop Service

```bash
sudo systemctl stop ornimetrics-detection
```

**Note:** Will still auto-start on next boot unless disabled.

---

### Start Service

```bash
sudo systemctl start ornimetrics-detection
```

---

### Restart Service

```bash
sudo systemctl restart ornimetrics-detection
```

---

### Disable Auto-Start

```bash
sudo systemctl disable ornimetrics-detection
```

**Result:** Will NOT start automatically on boot (but you can still start manually).

---

### Enable Auto-Start

```bash
sudo systemctl enable ornimetrics-detection
```

**Result:** Will start automatically on every boot.

---

## Crash Recovery

### Automatic Restart

If Ornimetrics OS crashes or exits unexpectedly:

```
1. Systemd detects the crash
2. Waits 10 seconds
3. Automatically restarts the service
4. System is back online
```

**No manual intervention needed.**

### Configuration

In `ornimetrics-detection.service`:

```ini
Restart=always       # Always restart on failure
RestartSec=10        # Wait 10 seconds before restart
```

---

## Power Loss Recovery

### What Happens

```
1. Power is lost → System shuts down
2. Power is restored → Raspberry Pi boots up
3. Systemd starts Ornimetrics OS automatically
4. System resumes normal operation
```

**Complete hands-free recovery.**

---

## Network Dependency

### Boot Sequence

Ornimetrics OS waits for network to be online before starting:

```ini
After=network-online.target
Wants=network-online.target
```

**This ensures:**
- WiFi is connected (if configured)
- Static IP is assigned
- Streaming is available
- Firebase can connect

**Boot time:** ~30-60 seconds depending on network

---

## Verification

### Check If Service Is Enabled

```bash
systemctl is-enabled ornimetrics-detection
```

**Expected output:** `enabled`

---

### Check If Service Is Running

```bash
systemctl is-active ornimetrics-detection
```

**Expected output:** `active`

---

### Check Boot Time

```bash
systemctl show ornimetrics-detection | grep ActiveEnterTimestamp
```

**Shows:** When service last started

---

## Troubleshooting

### Service Doesn't Start on Boot

**Check if enabled:**
```bash
systemctl is-enabled ornimetrics-detection
```

**If disabled:**
```bash
sudo systemctl enable ornimetrics-detection
```

---

### Service Crashes Immediately

**View error logs:**
```bash
sudo journalctl -u ornimetrics-detection -n 100 --no-pager
```

**Common issues:**
- Missing dependencies → Run OOBE again
- Config file errors → Check config_3d_detection.json
- Permission issues → Check file ownership

---

### Dashboard Not Accessible

**Check service status:**
```bash
sudo systemctl status ornimetrics-detection
```

**Check network:**
```bash
ip addr show
```

**Expected IP:** 192.168.1.200 (static)

**Try:**
- http://192.168.1.200:5000/
- http://raspberrypi.local:5000/
- http://{actual-ip}:5000/

---

## Files and Paths

### Service Files

```
/etc/systemd/system/ornimetrics-detection.service  # Systemd service file
/home/pi/Ornimetrics-AI/ornimetrics-detection.service  # Template
```

### Installation Scripts

```
/home/pi/Ornimetrics-AI/install_service.sh         # Interactive installer
/home/pi/Ornimetrics-AI/install_service_auto.sh    # Non-interactive installer
```

### Marker Files

```
/home/pi/Ornimetrics-AI/.oobe_completed           # Setup completion marker
```

### Log Files

```
# View with journalctl
sudo journalctl -u ornimetrics-detection -f
```

---

## Summary

### ✅ What You Get

- **Auto-start on every boot** - No manual intervention
- **Crash recovery** - Automatically restarts if it fails
- **Power loss recovery** - Resumes after power outage
- **Network awareness** - Waits for WiFi before starting
- **Zero configuration** - Works out of the box after OOBE

### 🎯 Your Experience

```
1. Power on device → Wait 60 seconds → Everything works
2. Power loss → Power restored → Wait 60 seconds → Everything works
3. System crash → Wait 10 seconds → Everything works
```

**That's it. Completely hands-free.**

---

## Advanced Configuration

### Change Restart Delay

Edit service file:
```bash
sudo nano /etc/systemd/system/ornimetrics-detection.service
```

Change:
```ini
RestartSec=10  # Change to desired seconds
```

Reload:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ornimetrics-detection
```

---

### Add Resource Limits

Edit service file and uncomment:
```ini
MemoryMax=2G      # Limit RAM usage
CPUQuota=200%     # Limit CPU usage
```

---

### Change Service Priority

Add to `[Service]` section:
```ini
Nice=-10  # Higher priority (-20 to 19, lower = higher priority)
```

---

## Support

**Check service status:**
```bash
sudo systemctl status ornimetrics-detection
```

**View logs:**
```bash
sudo journalctl -u ornimetrics-detection -n 100
```

**Restart service:**
```bash
sudo systemctl restart ornimetrics-detection
```

**Need help?**
- Check logs first
- Review configuration files
- Open GitHub issue

---

**Last Updated:** 2024-01-17
**Ornimetrics OS Version:** 1.0.0
**Status:** Production Ready
