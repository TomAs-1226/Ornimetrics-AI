# Web Detection Server Guide

## Overview

The Ornimetrics web detection server provides a complete web-based interface for bird detection with 3D individual recognition. It includes:

- **Live video stream** with detection overlays
- **Web dashboard** with real-time stats and controls
- **REST API** for integration with external apps
- **Auto-start on boot** capability
- **Hardware auto-detection**

## Quick Start

### 1. Install the Service (One-Time Setup)

```bash
cd /home/user/Ornimetrics-AI

# Install as systemd service (requires sudo)
sudo ./install_service.sh
```

This will:
- Install the systemd service
- Configure auto-start on boot
- Make all scripts executable
- Ask if you want to start now

### 2. Manual Start (Without Installing Service)

```bash
# Start the web server manually
./start_detection_system.sh

# Or with custom options
./start_detection_system.sh --port 8080 --host 0.0.0.0
```

### 3. Access the Dashboard

Once started, open your browser to:

```
http://raspberry-pi-ip:5000/
```

For example: `http://192.168.1.100:5000/`

## Features

### 🎥 Live Video Stream

The dashboard shows a live MJPEG stream with:
- Bounding boxes around detected birds
- Species labels and confidence scores
- Individual bird names (if 3D mode active)
- Real-time FPS counter
- System mode indicator (Full 3D or Optics-Only)

### 📊 Real-Time Statistics

- **FPS**: Current detection frame rate
- **Total Detections**: Birds detected since startup
- **Servo Triggers**: How many times food was dispensed
- **New Enrollments**: New individual birds registered

### 🐦 Individual Bird Tracking

When 3D mode is active, the dashboard shows:
- List of known individual birds
- Species for each bird
- Last seen timestamp
- Current status (Ready, Cooldown, etc.)

### 🎛️ Controls

- **Toggle Detection**: Pause/resume detection
- **Reset Database**: Clear all individual bird records

## REST API Endpoints

Your external app can use these endpoints:

### GET /api/status

Returns system status and hardware info.

**Response:**
```json
{
  "mode": "Full 3D",
  "backend": "Hailo-8 (26 TOPS)",
  "has_hailo": true,
  "has_3d": true,
  "has_individual_id": true,
  "detection_enabled": true,
  "uptime": 3600.5
}
```

### GET /api/stats

Returns detection statistics.

**Response:**
```json
{
  "fps": 11.8,
  "total_detections": 145,
  "total_triggers": 23,
  "total_enrollments": 8,
  "uptime_seconds": 3600.5
}
```

### GET /api/individuals

Returns list of known individual birds.

**Response:**
```json
{
  "individuals": [
    {
      "id": 1,
      "name": "Bird_1",
      "species": "cardinal",
      "last_seen": 1704067200.5,
      "can_dispense": false,
      "reason": "cooldown (120s remaining)"
    }
  ]
}
```

### GET /api/detections/recent

Returns recent detection history (last 100).

**Response:**
```json
{
  "detections": [
    {
      "time": "2024-01-01T12:00:00",
      "species": "cardinal",
      "confidence": 0.89,
      "individual_id": 1,
      "individual_name": "Bird_1"
    }
  ]
}
```

### POST /api/control/detection

Toggle detection on/off.

**Response:**
```json
{
  "enabled": true
}
```

### POST /api/control/reset_db

Reset individual bird database (requires confirmation).

**Response:**
```json
{
  "success": true,
  "message": "Database reset successfully"
}
```

### GET /video_feed

Returns MJPEG video stream. Can be embedded in any app:

```html
<img src="http://raspberry-pi-ip:5000/video_feed" />
```

## Auto-Start on Boot

### Install Service

```bash
sudo ./install_service.sh
```

Follow the prompts to:
1. Enable auto-start on boot
2. Start the service immediately

### Service Management

Once installed, manage the service with systemd:

```bash
# Check status
sudo systemctl status ornimetrics-detection

# Start service
sudo systemctl start ornimetrics-detection

# Stop service
sudo systemctl stop ornimetrics-detection

# Restart service
sudo systemctl restart ornimetrics-detection

# Enable auto-start on boot
sudo systemctl enable ornimetrics-detection

# Disable auto-start
sudo systemctl disable ornimetrics-detection

# View logs
sudo journalctl -u ornimetrics-detection -f

# View last 50 log lines
sudo journalctl -u ornimetrics-detection -n 50
```

### Verify Auto-Start

After enabling the service, reboot your Raspberry Pi:

```bash
sudo reboot
```

Wait for system to boot, then check if service started:

```bash
sudo systemctl status ornimetrics-detection
```

You should see "active (running)" in green.

## Configuration

Edit `config_3d_detection.json` to customize:

### Web Server Settings

```json
{
  "web": {
    "enabled": true,
    "host": "0.0.0.0",
    "port": 5000,
    "stream_fps": 15,
    "dashboard_enabled": true,
    "api_enabled": true
  }
}
```

- **host**: Bind address
  - `0.0.0.0` = accessible from any device on network
  - `127.0.0.1` = only accessible from Raspberry Pi itself
- **port**: Web server port (default 5000)
- **stream_fps**: Maximum FPS for video stream (15 recommended)

### After Changing Config

If service is installed:
```bash
sudo systemctl restart ornimetrics-detection
```

If running manually, stop and restart the script.

## Integration with Your App

### Option 1: Embed Video Stream

Add the MJPEG stream directly to your app:

```javascript
// React/React Native
<img src="http://192.168.1.100:5000/video_feed" />

// HTML
<img src="http://raspberry-pi-ip:5000/video_feed" alt="Bird Detection" />
```

### Option 2: Use REST API

Poll the API endpoints for data:

```javascript
// Fetch current stats
async function getStats() {
  const response = await fetch('http://192.168.1.100:5000/api/stats');
  const data = await response.json();
  console.log('FPS:', data.fps);
  console.log('Detections:', data.total_detections);
}

// Fetch individual birds
async function getBirds() {
  const response = await fetch('http://192.168.1.100:5000/api/individuals');
  const data = await response.json();
  return data.individuals;
}

// Poll every second
setInterval(getStats, 1000);
```

### Option 3: WebSocket (Future Enhancement)

Currently the API uses polling. For real-time updates, consider adding WebSocket support (not yet implemented).

## Network Access

### Local Network Only

By default, the server binds to `0.0.0.0:5000`, making it accessible from any device on your local network.

To find your Raspberry Pi's IP:
```bash
hostname -I
```

### Firewall Rules

If using a firewall, open port 5000:

```bash
sudo ufw allow 5000/tcp
```

### Remote Access

For access outside your local network, consider:

1. **Port Forwarding**: Forward port 5000 on your router
2. **VPN**: Use a VPN like WireGuard or OpenVPN
3. **Reverse Proxy**: Use ngrok or similar service

⚠️ **Security Warning**: Don't expose the server directly to the internet without authentication!

## Performance Tips

### For Best Streaming Performance

1. **Reduce stream FPS**: Set `stream_fps: 10` in config
2. **Use lighter backbone**: Set `backbone: "light"` for point cloud processing
3. **Process fewer frames**: Adjust detection FPS in config
4. **Use smaller resolution**: Reduce camera resolution if needed

### For Multiple Viewers

The server can handle multiple simultaneous viewers, but performance decreases with each connection. For many viewers, consider:

1. Using a dedicated streaming server (e.g., RTSP)
2. Reducing stream quality
3. Increasing hardware resources

## Troubleshooting

### Service Won't Start

Check logs:
```bash
sudo journalctl -u ornimetrics-detection -n 50
```

Common issues:
- **Python packages missing**: Run `pip3 install -r requirements.txt`
- **Model file not found**: Update `model_path` in config
- **Permission denied**: Ensure user has access to camera devices

### Can't Access Dashboard

1. **Check service is running**:
   ```bash
   sudo systemctl status ornimetrics-detection
   ```

2. **Check port is open**:
   ```bash
   sudo netstat -tlnp | grep 5000
   ```

3. **Check firewall**:
   ```bash
   sudo ufw status
   ```

4. **Try localhost first**:
   ```bash
   curl http://localhost:5000/api/status
   ```

### Video Stream Not Loading

1. **Check browser console** for errors
2. **Try different browser** (Chrome/Firefox recommended)
3. **Check camera permissions**: Ensure `/dev/video*` is accessible
4. **Reduce stream FPS** in config if network is slow

### High CPU Usage

1. **Use Hailo acceleration** if you have AI Hat
2. **Reduce detection FPS**: Process fewer frames
3. **Use light backbone**: Faster point cloud processing
4. **Disable 3D mode**: Use `--no-3d` flag or disable in config

## Development

### Running in Debug Mode

For development, you can run Flask in debug mode:

```python
# Edit web_detection_server.py
app.run(host=host, port=port, debug=True, threaded=True)
```

⚠️ Don't use debug mode in production!

### Adding Custom Endpoints

Edit `web_detection_server.py` and add new routes:

```python
@app.route('/api/custom')
def api_custom():
    return jsonify({"custom": "data"})
```

### Modifying Dashboard

The dashboard HTML is in `DASHBOARD_HTML` string in `web_detection_server.py`. Edit it to customize the UI.

## Security Considerations

### Default Setup

- No authentication required
- Accessible from any device on local network
- Suitable for trusted home networks

### Production Recommendations

1. **Add authentication**: Implement API keys or login
2. **Use HTTPS**: Set up SSL/TLS certificates
3. **Restrict access**: Change host to `127.0.0.1` and use reverse proxy
4. **Rate limiting**: Prevent API abuse
5. **Input validation**: Sanitize all user inputs

## Backup and Restore

### Backup Individual Bird Database

```bash
# Backup database
cp birdid.sqlite birdid.sqlite.backup

# Or with timestamp
cp birdid.sqlite "birdid.backup.$(date +%Y%m%d_%H%M%S).sqlite"
```

### Restore Database

```bash
# Stop service
sudo systemctl stop ornimetrics-detection

# Restore backup
cp birdid.sqlite.backup birdid.sqlite

# Start service
sudo systemctl start ornimetrics-detection
```

## Logs and Monitoring

### View Logs

```bash
# Real-time logs
sudo journalctl -u ornimetrics-detection -f

# Last hour
sudo journalctl -u ornimetrics-detection --since "1 hour ago"

# Errors only
sudo journalctl -u ornimetrics-detection -p err

# Export logs to file
sudo journalctl -u ornimetrics-detection > detection_logs.txt
```

### Log Rotation

Systemd automatically rotates logs. To configure:

```bash
sudo nano /etc/systemd/journald.conf

# Set these options:
SystemMaxUse=500M
MaxRetentionSec=7day
```

## Uninstalling

### Remove Service

```bash
# Stop and disable service
sudo systemctl stop ornimetrics-detection
sudo systemctl disable ornimetrics-detection

# Remove service file
sudo rm /etc/systemd/system/ornimetrics-detection.service

# Reload systemd
sudo systemctl daemon-reload
```

### Remove Database

```bash
rm birdid.sqlite
```

## Support

For issues or questions:
1. Check logs: `sudo journalctl -u ornimetrics-detection -n 50`
2. Verify configuration: `cat config_3d_detection.json`
3. Test manually: `./start_detection_system.sh`
4. Check hardware: Review detection script output

## Example Systemd Service Status

### Healthy Service
```
● ornimetrics-detection.service - Ornimetrics Bird Detection System
   Loaded: loaded (/etc/systemd/system/ornimetrics-detection.service; enabled)
   Active: active (running) since Mon 2024-01-01 12:00:00 GMT; 2h 30min ago
 Main PID: 12345 (python3)
   Status: "Running"
   CGroup: /system.slice/ornimetrics-detection.service
           └─12345 /usr/bin/python3 /home/pi/Ornimetrics-AI/web_detection_server.py

Jan 01 12:00:00 raspberrypi systemd[1]: Started Ornimetrics Bird Detection System.
Jan 01 12:00:01 raspberrypi python3[12345]: [INFO] Starting web server at http://0.0.0.0:5000
Jan 01 12:00:02 raspberrypi python3[12345]: [INFO] Dashboard: http://0.0.0.0:5000/
```

## That's It!

Your bird detection system is now:
- ✅ Running as a web server
- ✅ Accessible via browser dashboard
- ✅ Providing REST API for your app
- ✅ Auto-starting on boot
- ✅ Auto-detecting available hardware

Visit the dashboard to see it in action! 🐦
