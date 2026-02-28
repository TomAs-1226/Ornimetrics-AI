#!/usr/bin/env python3
"""
Ornimetrics OS - Bluetooth Pairing and Setup Service

Provides Bluetooth connectivity for mobile app to:
- Pair with the device
- Send WiFi credentials
- Link user account
- Configure feeder settings
- Initial setup without keyboard/monitor

Protocol: JSON over Bluetooth RFCOMM
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Callable
import hashlib
import secrets

logger = logging.getLogger(__name__)

try:
    import bluetooth
    BLUETOOTH_AVAILABLE = True
except ImportError:
    BLUETOOTH_AVAILABLE = False
    logger.warning("PyBluez not available - Bluetooth features disabled")


class BluetoothSetupService:
    """Bluetooth service for mobile app pairing and setup."""

    def __init__(self, config_path: str = "ornimetrics_os_config.json"):
        """Initialize Bluetooth service."""
        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.server_sock = None
        self.client_sock = None
        self.service_thread = None
        self.running = False

        # Pairing state
        self.paired = False
        self.pairing_token = None
        self.session_token = None

        # Callbacks for setup actions
        self.callbacks = {
            'wifi_configured': None,
            'account_linked': None,
            'settings_updated': None
        }

        # Service UUID (fixed for Ornimetrics OS)
        self.SERVICE_UUID = "00001101-0000-1000-8000-00805F9B34FB"

    def _load_config(self) -> Dict:
        """Load Ornimetrics OS configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_config(self):
        """Save configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def register_callback(self, event: str, callback: Callable):
        """Register callback for setup events."""
        if event in self.callbacks:
            self.callbacks[event] = callback

    def start(self):
        """Start Bluetooth service."""
        if not BLUETOOTH_AVAILABLE:
            logger.error("Bluetooth not available - cannot start service")
            return False

        if not self.config.get("bluetooth", {}).get("enabled", True):
            logger.info("Bluetooth disabled in config")
            return False

        self.running = True
        self.service_thread = threading.Thread(target=self._service_loop, daemon=True)
        self.service_thread.start()

        logger.info("Bluetooth service started")
        return True

    def stop(self):
        """Stop Bluetooth service."""
        self.running = False
        if self.client_sock:
            try:
                self.client_sock.close()
            except:
                pass
        if self.server_sock:
            try:
                self.server_sock.close()
            except:
                pass
        logger.info("Bluetooth service stopped")

    def _service_loop(self):
        """Main Bluetooth service loop."""
        try:
            # Create Bluetooth socket
            self.server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)

            # Bind to any available port
            self.server_sock.bind(("", bluetooth.PORT_ANY))
            self.server_sock.listen(1)

            port = self.server_sock.getsockname()[1]

            # Advertise service
            # Include device_id suffix so multiple feeders are distinguishable
            base_name = self.config.get("bluetooth", {}).get("device_name", "Ornimetrics OS")
            device_id = self.config.get("system", {}).get("device_id", "")
            id_suffix = device_id[-6:] if len(device_id) >= 6 else device_id
            device_name = f"{base_name}-{id_suffix}" if id_suffix else base_name

            bluetooth.advertise_service(
                self.server_sock,
                "Ornimetrics OS Setup",
                service_id=self.SERVICE_UUID,
                service_classes=[self.SERVICE_UUID, bluetooth.SERIAL_PORT_CLASS],
                profiles=[bluetooth.SERIAL_PORT_PROFILE],
                provider="Ornimetrics",
                description="Ornimetrics OS Mobile App Setup Service"
            )

            logger.info(f"Bluetooth service advertised on RFCOMM channel {port}")
            logger.info(f"Device name: {device_name}")
            logger.info("Waiting for mobile app connection...")

            while self.running:
                try:
                    # Accept connection (with timeout)
                    self.server_sock.settimeout(5.0)
                    client_sock, client_info = self.server_sock.accept()

                    logger.info(f"Accepted connection from {client_info}")
                    self.client_sock = client_sock

                    # Handle the connection
                    self._handle_client(client_sock, client_info)

                except bluetooth.BluetoothError as e:
                    if "timed out" not in str(e).lower():
                        logger.error(f"Bluetooth error: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error in service loop: {e}")
                    time.sleep(1)

        except Exception as e:
            logger.error(f"Failed to start Bluetooth service: {e}")
        finally:
            if self.server_sock:
                self.server_sock.close()

    def _handle_client(self, client_sock, client_info):
        """Handle client connection and commands."""
        try:
            # Send welcome message
            welcome = {
                "type": "welcome",
                "device_id": self.config.get("system", {}).get("device_id"),
                "device_name": self.config.get("system", {}).get("device_name"),
                "version": self.config.get("_branding", {}).get("version"),
                "requires_pairing": not self.paired
            }
            self._send_message(client_sock, welcome)

            # Command loop
            while self.running:
                try:
                    # Receive data
                    data = client_sock.recv(4096)
                    if not data:
                        break

                    # Parse command
                    try:
                        command = json.loads(data.decode('utf-8'))
                        response = self._process_command(command)
                        self._send_message(client_sock, response)
                    except json.JSONDecodeError:
                        self._send_error(client_sock, "Invalid JSON")

                except bluetooth.BluetoothError:
                    break

        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            client_sock.close()
            self.client_sock = None
            logger.info("Client disconnected")

    def _process_command(self, command: Dict) -> Dict:
        """Process command from mobile app."""
        cmd_type = command.get("type")

        if cmd_type == "pair":
            return self._handle_pair(command)
        elif cmd_type == "link_account":
            return self._handle_link_account(command)
        elif cmd_type == "configure_wifi":
            return self._handle_configure_wifi(command)
        elif cmd_type == "update_settings":
            return self._handle_update_settings(command)
        elif cmd_type == "get_status":
            return self._handle_get_status(command)
        else:
            return {"type": "error", "message": f"Unknown command: {cmd_type}"}

    def _handle_pair(self, command: Dict) -> Dict:
        """Handle pairing request from app."""
        # Generate pairing token
        self.pairing_token = secrets.token_hex(32)

        # Generate session token
        self.session_token = secrets.token_hex(32)

        # Mark as paired
        self.paired = True

        # Add to paired devices
        if "paired_devices" not in self.config.get("bluetooth", {}):
            self.config["bluetooth"]["paired_devices"] = []

        device_info = {
            "app_id": command.get("app_id"),
            "device_model": command.get("device_model"),
            "paired_at": time.time()
        }
        self.config["bluetooth"]["paired_devices"].append(device_info)
        self._save_config()

        logger.info(f"Paired with mobile app: {device_info}")

        return {
            "type": "pair_success",
            "session_token": self.session_token,
            "device_id": self.config["system"]["device_id"]
        }

    def _handle_link_account(self, command: Dict) -> Dict:
        """Link device to user account."""
        if not self._verify_session(command):
            return {"type": "error", "message": "Not authenticated"}

        user_id = command.get("user_id")
        account_email = command.get("account_email")
        account_token = command.get("account_token")

        if not user_id or not account_email:
            return {"type": "error", "message": "Missing user_id or account_email"}

        # Update account info
        self.config["account"] = {
            "linked": True,
            "user_id": user_id,
            "account_email": account_email,
            "account_token": account_token,
            "linked_timestamp": time.time(),
            "feeder_name": command.get("feeder_name", "My Feeder"),
            "sharing_enabled": False
        }

        self._save_config()

        # Callback
        if self.callbacks['account_linked']:
            self.callbacks['account_linked'](user_id, account_email)

        logger.info(f"Account linked: {account_email} (User ID: {user_id})")

        return {
            "type": "account_linked",
            "user_id": user_id,
            "device_id": self.config["system"]["device_id"]
        }

    def _handle_configure_wifi(self, command: Dict) -> Dict:
        """Configure WiFi credentials."""
        if not self._verify_session(command):
            return {"type": "error", "message": "Not authenticated"}

        ssid = command.get("ssid")
        password = command.get("password")

        if not ssid:
            return {"type": "error", "message": "Missing SSID"}

        # Update config
        self.config["network"]["wifi_configured"] = True
        self.config["network"]["wifi_ssid"] = ssid
        self._save_config()

        # Callback to actually configure WiFi
        if self.callbacks['wifi_configured']:
            success = self.callbacks['wifi_configured'](ssid, password)
            if not success:
                return {"type": "error", "message": "Failed to configure WiFi"}

        logger.info(f"WiFi configured: {ssid}")

        return {
            "type": "wifi_configured",
            "ssid": ssid,
            "static_ip": self.config["network"].get("static_ip")
        }

    def _handle_update_settings(self, command: Dict) -> Dict:
        """Update feeder settings."""
        if not self._verify_session(command):
            return {"type": "error", "message": "Not authenticated"}

        settings = command.get("settings", {})

        # Update allowed settings
        if "device_name" in settings:
            self.config["system"]["device_name"] = settings["device_name"]

        if "feeder_name" in settings:
            self.config["account"]["feeder_name"] = settings["feeder_name"]

        if "features" in settings:
            for feature, enabled in settings["features"].items():
                if feature in self.config.get("features", {}):
                    self.config["features"][feature]["enabled"] = enabled

        self._save_config()

        # Callback
        if self.callbacks['settings_updated']:
            self.callbacks['settings_updated'](settings)

        logger.info("Settings updated via Bluetooth")

        return {
            "type": "settings_updated",
            "success": True
        }

    def _handle_get_status(self, command: Dict) -> Dict:
        """Get current device status."""
        return {
            "type": "status",
            "device_id": self.config["system"]["device_id"],
            "device_name": self.config["system"]["device_name"],
            "version": self.config["_branding"]["version"],
            "account_linked": self.config["account"]["linked"],
            "wifi_configured": self.config["network"]["wifi_configured"],
            "static_ip": self.config["network"].get("static_ip"),
            "streaming": {
                "enabled": self.config["streaming"]["enabled"],
                "mjpeg_url": f"http://{self.config['network']['static_ip']}:5000/video_feed",
                "rtsp_url": f"rtsp://{self.config['network']['static_ip']}:8554/ornimetrics/stream"
            }
        }

    def _verify_session(self, command: Dict) -> bool:
        """Verify session token."""
        token = command.get("session_token")
        return token == self.session_token

    def _send_message(self, sock, message: Dict):
        """Send JSON message over Bluetooth."""
        data = json.dumps(message).encode('utf-8')
        sock.send(data + b'\n')

    def _send_error(self, sock, error: str):
        """Send error message."""
        self._send_message(sock, {"type": "error", "message": error})

    def get_pairing_info(self) -> Dict:
        """Get info for displaying pairing instructions."""
        return {
            "device_name": self.config.get("bluetooth", {}).get("device_name", "Ornimetrics OS"),
            "service_name": "Ornimetrics OS Setup",
            "pin_required": self.config.get("bluetooth", {}).get("pin_required", True),
            "pin_code": self.config.get("bluetooth", {}).get("pin_code", "1234"),
            "paired": self.paired
        }


def configure_wifi_callback(ssid: str, password: str) -> bool:
    """Callback to configure WiFi with given credentials."""
    try:
        # Create wpa_supplicant configuration
        wpa_conf = f"""
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=US

network={{
    ssid="{ssid}"
    psk="{password}"
    key_mgmt=WPA-PSK
}}
"""

        # Write to wpa_supplicant.conf
        with open('/etc/wpa_supplicant/wpa_supplicant.conf', 'w') as f:
            f.write(wpa_conf)

        # Restart networking
        import subprocess
        subprocess.run(['sudo', 'systemctl', 'restart', 'networking'], check=True)
        subprocess.run(['sudo', 'wpa_cli', '-i', 'wlan0', 'reconfigure'], check=False)

        logger.info(f"WiFi configured for SSID: {ssid}")
        return True

    except Exception as e:
        logger.error(f"Failed to configure WiFi: {e}")
        return False


if __name__ == "__main__":
    # Test Bluetooth service
    logging.basicConfig(level=logging.INFO)

    service = BluetoothSetupService()
    service.register_callback('wifi_configured', configure_wifi_callback)

    if service.start():
        print("Bluetooth service running. Waiting for mobile app...")
        print(f"Pairing info: {service.get_pairing_info()}")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            service.stop()
    else:
        print("Failed to start Bluetooth service")
