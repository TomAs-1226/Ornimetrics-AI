# Ornimetrics OS - Mobile App Developer Guide

Complete guide for developing mobile applications to integrate with Ornimetrics OS bird feeders.

---

## Table of Contents

1. [Overview](#overview)
2. [Bluetooth Pairing Protocol](#bluetooth-pairing-protocol)
3. [REST API Reference](#rest-api-reference)
4. [Video Streaming](#video-streaming)
5. [Firebase Data Structure](#firebase-data-structure)
6. [Setup Workflow](#setup-workflow)
7. [Code Examples](#code-examples)

---

## Overview

Ornimetrics OS provides multiple interfaces for mobile app integration:

- **Bluetooth** - Initial device pairing, WiFi setup, account linking
- **REST API** - Real-time status, control, and data access
- **MJPEG/RTSP** - Live video streaming
- **Firebase** - Cloud data synchronization (user-based)

### Key Features

- Headless setup via Bluetooth (no keyboard/monitor needed)
- Secure account-based data isolation
- Local network streaming (static IP)
- Real-time detection notifications
- Individual bird recognition

---

## Bluetooth Pairing Protocol

### Service Discovery

**Service Name:** `Ornimetrics OS Setup`
**Service UUID:** `00001101-0000-1000-8000-00805F9B34FB`
**Protocol:** RFCOMM
**Port:** Any available port (advertised via SDP)

### Device Name Format

```
Ornimetrics-{hostname}
```

Example: `Ornimetrics-raspberrypi`

### Connection Flow

```
1. Scan for Bluetooth devices
2. Filter by service UUID or name prefix "Ornimetrics-"
3. Connect to RFCOMM socket
4. Receive welcome message
5. Send commands as JSON
6. Receive responses as JSON
```

### Message Format

All messages are JSON objects terminated with newline (`\n`).

**Welcome Message (received on connect):**

```json
{
  "type": "welcome",
  "device_id": "uuid-string",
  "device_name": "My Feeder",
  "version": "1.0.0",
  "requires_pairing": true
}
```

---

## Bluetooth Commands

### 1. Pair Command

Establish pairing session with device.

**Request:**
```json
{
  "type": "pair",
  "app_id": "ornimetrics-mobile-app",
  "device_model": "iPhone 13",
  "app_version": "1.0.0"
}
```

**Response (Success):**
```json
{
  "type": "pair_success",
  "session_token": "hex-token-string",
  "device_id": "uuid-string"
}
```

**Note:** Save `session_token` for subsequent commands.

---

### 2. Link Account Command

Link device to user's Ornimetrics account.

**Request:**
```json
{
  "type": "link_account",
  "session_token": "hex-token-from-pair",
  "user_id": "firebase-user-id",
  "account_email": "user@example.com",
  "account_token": "firebase-id-token",
  "feeder_name": "Backyard Feeder"
}
```

**Response (Success):**
```json
{
  "type": "account_linked",
  "user_id": "firebase-user-id",
  "device_id": "uuid-string"
}
```

**Response (Error):**
```json
{
  "type": "error",
  "message": "Not authenticated"
}
```

**Note:** After successful account linking, device will start publishing data to user's Firebase path.

---

### 3. Configure WiFi Command

Send WiFi credentials to device.

**Request:**
```json
{
  "type": "configure_wifi",
  "session_token": "hex-token-from-pair",
  "ssid": "YourNetworkName",
  "password": "YourNetworkPassword"
}
```

**Response (Success):**
```json
{
  "type": "wifi_configured",
  "ssid": "YourNetworkName",
  "static_ip": "192.168.1.200"
}
```

**Response (Error):**
```json
{
  "type": "error",
  "message": "Failed to configure WiFi"
}
```

**Note:** After WiFi configuration, device will have IP address (static: `192.168.1.200` by default).

---

### 4. Update Settings Command

Update device settings.

**Request:**
```json
{
  "type": "update_settings",
  "session_token": "hex-token-from-pair",
  "settings": {
    "device_name": "Front Yard Feeder",
    "feeder_name": "Bird Station #1",
    "features": {
      "individual_recognition": true,
      "auto_training": true,
      "3d_camera": true
    }
  }
}
```

**Response (Success):**
```json
{
  "type": "settings_updated",
  "success": true
}
```

---

### 5. Get Status Command

Get current device status.

**Request:**
```json
{
  "type": "get_status"
}
```

**Response:**
```json
{
  "type": "status",
  "device_id": "uuid-string",
  "device_name": "My Feeder",
  "version": "1.0.0",
  "account_linked": true,
  "wifi_configured": true,
  "static_ip": "192.168.1.200",
  "streaming": {
    "enabled": true,
    "mjpeg_url": "http://192.168.1.200:5000/video_feed",
    "rtsp_url": "rtsp://192.168.1.200:8554/ornimetrics/stream"
  }
}
```

---

## REST API Reference

All API endpoints are available at `http://{device-ip}:5000/api/`.

### Authentication

Currently no authentication required for local network access. Future versions may add token-based auth.

---

### GET /api/status

Get system status.

**Response:**
```json
{
  "system": {
    "uptime_seconds": 3600,
    "device_id": "uuid-string",
    "version": "1.0.0",
    "account_linked": true
  },
  "hardware": {
    "depth_camera_available": true,
    "hailo_available": true,
    "mode": "Full 3D Mode"
  },
  "detection": {
    "active": true,
    "fps": 15.3,
    "total_detections": 127
  }
}
```

---

### GET /api/stats

Get detection statistics.

**Response:**
```json
{
  "total_detections": 127,
  "species_counts": {
    "Cardinal": 45,
    "Blue_Jay": 32,
    "House_Finch": 20
  },
  "unique_individuals": 15,
  "detections_today": 34,
  "last_detection_time": "2024-01-17T14:35:22"
}
```

---

### GET /api/individuals

Get known individual birds.

**Response:**
```json
{
  "individuals": [
    {
      "id": 1,
      "species": "Cardinal",
      "name": "Cardinal #1",
      "first_seen": "2024-01-15T08:30:00",
      "last_seen": "2024-01-17T14:20:15",
      "visit_count": 23,
      "confidence": 0.92
    }
  ]
}
```

---

### GET /api/recent_detections

Get recent detections (last 50).

**Query Parameters:**
- `limit` (optional): Number of detections to return (default: 50, max: 200)
- `species` (optional): Filter by species name
- `individual_id` (optional): Filter by individual ID

**Response:**
```json
{
  "detections": [
    {
      "timestamp": "2024-01-17T14:35:22",
      "species": "Cardinal",
      "confidence": 0.89,
      "individual_id": 3,
      "bbox": [100, 150, 300, 400],
      "has_3d": true
    }
  ]
}
```

---

### GET /api/training/status

Get training system status (if enabled).

**Response:**
```json
{
  "training_enabled": true,
  "is_training": false,
  "night_mode_active": false,
  "training_status": {
    "status": "idle",
    "progress": 0.0,
    "message": "Waiting for scheduled time"
  },
  "dataset_ready": true,
  "species_counts": {
    "Cardinal": 127,
    "Blue_Jay": 89
  },
  "last_training_time": "2024-01-17T02:30:00"
}
```

---

### POST /api/training/start

Manually trigger training.

**Response:**
```json
{
  "success": true,
  "message": "Training started"
}
```

---

### POST /api/training/rollback

Rollback to previous model.

**Response:**
```json
{
  "success": true,
  "message": "Rolled back to previous model"
}
```

---

## Video Streaming

### MJPEG Stream

**URL:** `http://{device-ip}:5000/video_feed`

**Format:** Motion JPEG
**Protocol:** HTTP
**Latency:** Low (< 200ms)

**Usage in iOS (Swift):**
```swift
let url = URL(string: "http://192.168.1.200:5000/video_feed")!
let imageView = UIImageView()
// Use AsyncImageView or custom MJPEG decoder
```

**Usage in Android (Kotlin):**
```kotlin
val url = "http://192.168.1.200:5000/video_feed"
val imageView = findViewById<ImageView>(R.id.stream)
// Use MjpegInputStream library
```

---

### RTSP Stream

**URL:** `rtsp://{device-ip}:8554/ornimetrics/stream`

**Format:** H.264
**Protocol:** RTSP
**Latency:** Higher (500-1000ms)
**Quality:** Better compression

**Usage (iOS):**
```swift
import AVKit

let url = URL(string: "rtsp://192.168.1.200:8554/ornimetrics/stream")!
let player = AVPlayer(url: url)
let playerLayer = AVPlayerLayer(player: player)
// Add to view hierarchy
player.play()
```

**Usage (Android):**
```kotlin
import android.net.Uri
import com.google.android.exoplayer2.ExoPlayer

val uri = Uri.parse("rtsp://192.168.1.200:8554/ornimetrics/stream")
val player = ExoPlayer.Builder(context).build()
player.setMediaItem(MediaItem.fromUri(uri))
player.prepare()
player.play()
```

---

## Firebase Data Structure

All user data is stored under user-based paths for privacy and isolation.

### Path Template

```
/users/{user_id}/feeders/{device_id}/
```

### Collections

#### Detections

Path: `/users/{user_id}/feeders/{device_id}/detections/`

Document structure:
```json
{
  "timestamp": "2024-01-17T14:35:22.123Z",
  "species": "Cardinal",
  "yolo_confidence": 0.89,
  "individual_id": 3,
  "individual_confidence": 0.92,
  "bbox": [100, 150, 300, 400],
  "has_3d": true,
  "image_url": "gs://bucket/users/{user_id}/feeders/{device_id}/images/...",
  "pointcloud_url": "gs://bucket/users/{user_id}/feeders/{device_id}/pointclouds/...",
  "_ornimetrics_os": {
    "version": "1.0.0",
    "device_id": "uuid-string",
    "feeder_name": "My Feeder"
  }
}
```

#### Individuals

Path: `/users/{user_id}/feeders/{device_id}/individuals/`

Document structure:
```json
{
  "id": 3,
  "species": "Cardinal",
  "name": "Cardinal #3",
  "first_seen": "2024-01-15T08:30:00Z",
  "last_seen": "2024-01-17T14:35:22Z",
  "visit_count": 23,
  "average_confidence": 0.92,
  "prototype_embedding": [/* 512-dim array */]
}
```

#### Statistics

Path: `/users/{user_id}/feeders/{device_id}/statistics/daily/{date}`

Document structure:
```json
{
  "date": "2024-01-17",
  "total_detections": 34,
  "unique_individuals": 8,
  "species_breakdown": {
    "Cardinal": 12,
    "Blue_Jay": 10,
    "House_Finch": 12
  },
  "busiest_hour": "08:00",
  "peak_detections": 8
}
```

---

## Setup Workflow

Complete workflow for setting up a new Ornimetrics OS feeder via mobile app.

### Step-by-Step Flow

```
1. User Powers On Device
   └─> Device starts in OOBE mode (no account linked)
   └─> Bluetooth service starts automatically
   └─> Device advertises as "Ornimetrics-{hostname}"

2. App: Scan for Devices
   └─> User taps "Add New Feeder"
   └─> App scans for Bluetooth devices
   └─> Filter by service UUID or name prefix
   └─> Show list of available feeders

3. App: Connect to Device
   └─> User selects device from list
   └─> App connects to RFCOMM socket
   └─> Receive welcome message
   └─> Check requires_pairing flag

4. App: Pair with Device
   └─> Send "pair" command with app info
   └─> Receive session_token
   └─> Store token for subsequent commands

5. App: Link Account
   └─> User logs in to Ornimetrics account (Firebase Auth)
   └─> Send "link_account" command with user_id, email, token
   └─> Device updates configuration
   └─> Device creates user-based Firebase path

6. App: Configure WiFi
   └─> User enters WiFi credentials
   └─> Send "configure_wifi" command
   └─> Device configures wpa_supplicant
   └─> Device connects to WiFi
   └─> Device assigns static IP (192.168.1.200)

7. App: Get Status
   └─> Send "get_status" command
   └─> Receive streaming URLs (MJPEG, RTSP)
   └─> Store device_id and IP address

8. App: Switch to Local Network
   └─> Disconnect Bluetooth
   └─> Connect to WiFi network (same as device)
   └─> Access device via REST API (http://192.168.1.200:5000)
   └─> Start video stream
   └─> Show live detections

9. Device: Complete Setup
   └─> Device marks OOBE as completed
   └─> Device starts detection system
   └─> Device publishes data to Firebase
   └─> User sees data in app (via Firebase listeners)
```

---

## Code Examples

### iOS (Swift) - Complete Setup Flow

```swift
import CoreBluetooth
import Firebase

class OrnimetricsSetup: NSObject, CBCentralManagerDelegate, CBPeripheralDelegate {
    var centralManager: CBCentralManager!
    var feederPeripheral: CBPeripheral?
    var socket: StreamDelegate?
    var sessionToken: String?

    // Service UUID
    let serviceUUID = CBUUID(string: "00001101-0000-1000-8000-00805F9B34FB")

    // 1. Scan for devices
    func startScanning() {
        centralManager = CBCentralManager(delegate: self, queue: nil)
    }

    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        if central.state == .poweredOn {
            centralManager.scanForPeripherals(withServices: [serviceUUID], options: nil)
        }
    }

    func centralManager(_ central: CBCentralManager,
                       didDiscover peripheral: CBPeripheral,
                       advertisementData: [String : Any],
                       rssi RSSI: NSNumber) {
        if let name = peripheral.name, name.starts(with: "Ornimetrics-") {
            print("Found feeder: \(name)")
            feederPeripheral = peripheral
            centralManager.stopScan()
            centralManager.connect(peripheral, options: nil)
        }
    }

    // 2. Connect and pair
    func centralManager(_ central: CBCentralManager,
                       didConnect peripheral: CBPeripheral) {
        peripheral.delegate = self
        peripheral.discoverServices([serviceUUID])

        // Send pair command
        let pairCommand = [
            "type": "pair",
            "app_id": "ornimetrics-ios-app",
            "device_model": UIDevice.current.model,
            "app_version": "1.0.0"
        ]

        sendCommand(pairCommand)
    }

    // 3. Link account
    func linkAccount(userId: String, email: String, token: String, feederName: String) {
        let linkCommand = [
            "type": "link_account",
            "session_token": sessionToken!,
            "user_id": userId,
            "account_email": email,
            "account_token": token,
            "feeder_name": feederName
        ]

        sendCommand(linkCommand)
    }

    // 4. Configure WiFi
    func configureWiFi(ssid: String, password: String) {
        let wifiCommand = [
            "type": "configure_wifi",
            "session_token": sessionToken!,
            "ssid": ssid,
            "password": password
        ]

        sendCommand(wifiCommand)
    }

    // 5. Get streaming URLs
    func getStatus() {
        let statusCommand = ["type": "get_status"]
        sendCommand(statusCommand)
    }

    func sendCommand(_ command: [String: Any]) {
        guard let jsonData = try? JSONSerialization.data(withJSONObject: command),
              var jsonString = String(data: jsonData, encoding: .utf8) else {
            return
        }

        jsonString += "\n"
        // Send via Bluetooth socket
        socket?.send(jsonString)
    }
}
```

---

### Android (Kotlin) - REST API Integration

```kotlin
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.*

// API Interface
interface OrnimetricsAPI {
    @GET("api/status")
    suspend fun getStatus(): SystemStatus

    @GET("api/stats")
    suspend fun getStats(): DetectionStats

    @GET("api/individuals")
    suspend fun getIndividuals(): IndividualsList

    @GET("api/recent_detections")
    suspend fun getRecentDetections(
        @Query("limit") limit: Int = 50,
        @Query("species") species: String? = null
    ): DetectionsList

    @POST("api/training/start")
    suspend fun startTraining(): ApiResponse
}

// Data classes
data class SystemStatus(
    val system: SystemInfo,
    val hardware: HardwareInfo,
    val detection: DetectionInfo
)

data class SystemInfo(
    val uptime_seconds: Int,
    val device_id: String,
    val version: String,
    val account_linked: Boolean
)

// Usage
class FeederRepository {
    private val api: OrnimetricsAPI by lazy {
        Retrofit.Builder()
            .baseUrl("http://192.168.1.200:5000/")
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(OrnimetricsAPI::class.java)
    }

    suspend fun getFeederStatus(): SystemStatus {
        return api.getStatus()
    }

    suspend fun getDetectionStats(): DetectionStats {
        return api.getStats()
    }
}

// ViewModel
class FeederViewModel(private val repository: FeederRepository) : ViewModel() {
    val status = MutableLiveData<SystemStatus>()
    val stats = MutableLiveData<DetectionStats>()

    fun refreshData() {
        viewModelScope.launch {
            try {
                status.value = repository.getFeederStatus()
                stats.value = repository.getDetectionStats()
            } catch (e: Exception) {
                // Handle error
            }
        }
    }
}
```

---

### Firebase Listeners (iOS/Swift)

```swift
import Firebase

class FeederDataManager {
    let db = Firestore.firestore()
    var listener: ListenerRegistration?

    func listenToDetections(userId: String, deviceId: String) {
        let path = "users/\(userId)/feeders/\(deviceId)/detections"

        listener = db.collection(path)
            .order(by: "timestamp", descending: true)
            .limit(to: 50)
            .addSnapshotListener { snapshot, error in
                guard let documents = snapshot?.documents else {
                    print("Error: \(error?.localizedDescription ?? "unknown")")
                    return
                }

                let detections = documents.compactMap { doc -> Detection? in
                    try? doc.data(as: Detection.self)
                }

                // Update UI with detections
                self.updateUI(detections)
            }
    }

    func getIndividuals(userId: String, deviceId: String) async -> [Individual] {
        let path = "users/\(userId)/feeders/\(deviceId)/individuals"

        let snapshot = try? await db.collection(path).getDocuments()
        return snapshot?.documents.compactMap {
            try? $0.data(as: Individual.self)
        } ?? []
    }
}

struct Detection: Codable {
    let timestamp: String
    let species: String
    let yolo_confidence: Double
    let individual_id: Int?
    let bbox: [Int]
    let has_3d: Bool
}

struct Individual: Codable {
    let id: Int
    let species: String
    let name: String
    let first_seen: String
    let last_seen: String
    let visit_count: Int
    let average_confidence: Double
}
```

---

## Security Considerations

### Bluetooth Pairing

- Session tokens are randomly generated (256-bit hex)
- Tokens expire when Bluetooth disconnects
- Only one pairing session allowed at a time
- No PIN required by default (can be enabled in config)

### Local Network Access

- REST API currently has no authentication (local network only)
- Static IP ensures consistent access
- Future versions may add token-based auth

### Firebase Data

- User-based paths ensure data isolation
- Each user only sees their own feeders
- Firebase Security Rules should enforce:
  ```
  match /users/{userId}/feeders/{deviceId}/{document=**} {
    allow read, write: if request.auth.uid == userId;
  }
  ```

### Privacy

- All bird detection data is private to user account
- No public sharing by default
- Images and point clouds stored in user-specific paths
- Device only uploads data when account is linked

---

## Troubleshooting

### Bluetooth Connection Issues

**Problem:** Can't find device in Bluetooth scan

**Solutions:**
- Ensure device is powered on and in OOBE mode (no `.oobe_completed` file)
- Check Bluetooth is enabled on mobile device
- Try restarting device
- Check device name in system logs

---

**Problem:** Connection drops during setup

**Solutions:**
- Stay within Bluetooth range (< 10m)
- Avoid interference from other devices
- Retry pairing process
- Check device logs for errors

---

### WiFi Configuration Issues

**Problem:** Device doesn't connect to WiFi after configuration

**Solutions:**
- Verify SSID and password are correct
- Check WiFi network is 2.4GHz (5GHz may not be supported)
- Ensure network uses WPA/WPA2 security
- Check device logs: `sudo journalctl -u ornimetrics-detection`

---

### Streaming Issues

**Problem:** Can't access video stream

**Solutions:**
- Verify device and mobile app are on same WiFi network
- Check static IP is correct (default: 192.168.1.200)
- Try MJPEG stream first (lower latency)
- Check firewall rules on router
- Verify port 5000 is accessible

---

### API Connection Issues

**Problem:** REST API returns errors or timeouts

**Solutions:**
- Verify device IP and port (default: http://192.168.1.200:5000)
- Check device is online: `ping 192.168.1.200`
- Verify detection system is running: `sudo systemctl status ornimetrics-detection`
- Check API endpoint paths are correct
- Review device logs for errors

---

## Support

For development support:
- GitHub Issues: https://github.com/ornimetrics/ornimetrics-os/issues
- Documentation: See all `*.md` files in project root
- Email: dev@ornimetrics.com

---

**Last Updated:** 2024-01-17
**API Version:** 1.0.0
**Ornimetrics OS Version:** 1.0.0
