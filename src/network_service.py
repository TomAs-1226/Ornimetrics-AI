#!/usr/bin/env python3
"""
Ornimetrics OS - Network Configuration Service

Manages network configuration for Ornimetrics OS:
- Sets and enforces static IP address
- Configures WiFi credentials
- Ensures consistent network setup for mobile app connectivity
- Handles both Ethernet and WiFi interfaces

The static IP (default 192.168.1.200) ensures the mobile app can reliably
find the streaming endpoints without discovery protocols.
"""

import subprocess
import logging
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class NetworkConfigService:
    """Manages network configuration for Ornimetrics OS."""

    def __init__(self, config_path: str = "ornimetrics_os_config.json"):
        """Initialize network service."""
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Network settings
        self.static_ip_enabled = self.config.get("network", {}).get("static_ip_enabled", True)
        self.static_ip = self.config.get("network", {}).get("static_ip", "192.168.1.200")
        self.subnet_mask = self.config.get("network", {}).get("subnet_mask", "255.255.255.0")
        self.gateway = self.config.get("network", {}).get("gateway", "192.168.1.1")
        self.dns_servers = self.config.get("network", {}).get("dns_servers", ["8.8.8.8", "8.8.4.4"])

        # Interface names
        self.eth_interface = "eth0"
        self.wlan_interface = "wlan0"

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

    def get_active_interface(self) -> Optional[str]:
        """Determine which network interface is active."""
        try:
            # Check Ethernet first
            result = subprocess.run(
                ['ip', 'link', 'show', self.eth_interface],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and 'state UP' in result.stdout:
                return self.eth_interface

            # Check WiFi
            result = subprocess.run(
                ['ip', 'link', 'show', self.wlan_interface],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and 'state UP' in result.stdout:
                return self.wlan_interface

            return None

        except Exception as e:
            logger.error(f"Error detecting active interface: {e}")
            return None

    def get_current_ip(self, interface: Optional[str] = None) -> Optional[str]:
        """Get current IP address of interface."""
        if not interface:
            interface = self.get_active_interface()

        if not interface:
            return None

        try:
            result = subprocess.run(
                ['ip', '-4', 'addr', 'show', interface],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # Parse IP from output
                for line in result.stdout.split('\n'):
                    if 'inet ' in line:
                        ip = line.strip().split()[1].split('/')[0]
                        return ip

            return None

        except Exception as e:
            logger.error(f"Error getting current IP: {e}")
            return None

    def configure_static_ip_dhcpcd(self, interface: str) -> bool:
        """Configure static IP using dhcpcd (Raspberry Pi OS default)."""
        try:
            dhcpcd_conf = Path('/etc/dhcpcd.conf')

            # Check if we can write to the file
            if not dhcpcd_conf.exists():
                logger.error("dhcpcd.conf not found")
                return False

            # Read existing config
            existing_config = dhcpcd_conf.read_text()

            # Remove any existing static IP config for this interface
            lines = existing_config.split('\n')
            new_lines = []
            skip_until_next_interface = False

            for line in lines:
                if line.startswith('interface '):
                    skip_until_next_interface = interface in line
                elif skip_until_next_interface and line.strip() and not line.startswith('interface '):
                    continue  # Skip lines in our interface block
                else:
                    skip_until_next_interface = False
                    new_lines.append(line)

            # Add our static IP configuration
            static_config = f"""
# Ornimetrics OS Static IP Configuration
interface {interface}
static ip_address={self.static_ip}/24
static routers={self.gateway}
static domain_name_servers={' '.join(self.dns_servers)}
"""

            new_config = '\n'.join(new_lines) + '\n' + static_config

            # Write config (requires sudo)
            temp_file = Path('/tmp/dhcpcd.conf.ornimetrics')
            temp_file.write_text(new_config)

            subprocess.run(
                ['sudo', 'cp', str(temp_file), '/etc/dhcpcd.conf'],
                check=True,
                timeout=10
            )

            temp_file.unlink()

            logger.info(f"Static IP configured for {interface}: {self.static_ip}")
            return True

        except Exception as e:
            logger.error(f"Failed to configure static IP: {e}")
            return False

    def apply_network_configuration(self) -> bool:
        """Apply network configuration and restart networking."""
        try:
            # Restart dhcpcd service
            subprocess.run(
                ['sudo', 'systemctl', 'restart', 'dhcpcd'],
                check=True,
                timeout=30
            )

            logger.info("Network configuration applied")

            # Wait for network to stabilize
            time.sleep(5)

            # Verify IP
            current_ip = self.get_current_ip()
            if current_ip == self.static_ip:
                logger.info(f"Static IP verified: {current_ip}")
                self.config["network"]["current_ip"] = current_ip
                self.config["network"]["configured"] = True
                self._save_config()
                return True
            else:
                logger.warning(f"IP mismatch: expected {self.static_ip}, got {current_ip}")
                return False

        except Exception as e:
            logger.error(f"Failed to apply network configuration: {e}")
            return False

    def configure_wifi(self, ssid: str, password: str) -> bool:
        """Configure WiFi credentials using wpa_supplicant."""
        try:
            # Create wpa_supplicant configuration
            wpa_conf = f"""ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=US

network={{
    ssid="{ssid}"
    psk="{password}"
    key_mgmt=WPA-PSK
}}
"""

            # Write to temp file
            temp_file = Path('/tmp/wpa_supplicant.conf.ornimetrics')
            temp_file.write_text(wpa_conf)

            # Copy to system location (requires sudo)
            subprocess.run(
                ['sudo', 'cp', str(temp_file), '/etc/wpa_supplicant/wpa_supplicant.conf'],
                check=True,
                timeout=10
            )

            temp_file.unlink()

            # Reconfigure wpa_supplicant
            subprocess.run(
                ['sudo', 'wpa_cli', '-i', self.wlan_interface, 'reconfigure'],
                check=False,  # May fail if interface not ready
                timeout=10
            )

            # Save to config
            self.config["network"]["wifi_configured"] = True
            self.config["network"]["wifi_ssid"] = ssid
            self._save_config()

            logger.info(f"WiFi configured for SSID: {ssid}")

            # Wait for connection
            time.sleep(10)

            # Verify WiFi connection
            if self._is_wifi_connected():
                logger.info("WiFi connection successful")
                return True
            else:
                logger.warning("WiFi configured but connection not verified")
                return True  # Still return True as config was written

        except Exception as e:
            logger.error(f"Failed to configure WiFi: {e}")
            return False

    def _is_wifi_connected(self) -> bool:
        """Check if WiFi is connected."""
        try:
            result = subprocess.run(
                ['iwconfig', self.wlan_interface],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                return 'ESSID:' in result.stdout and 'off/any' not in result.stdout

            return False

        except Exception as e:
            logger.debug(f"Could not check WiFi status: {e}")
            return False

    def setup_network(self) -> Tuple[bool, str]:
        """Complete network setup process."""
        logger.info("Starting network setup...")

        # Determine active interface
        interface = self.get_active_interface()

        if not interface:
            logger.error("No active network interface found")
            return False, "No active network interface"

        logger.info(f"Active interface: {interface}")

        # Configure static IP if enabled
        if self.static_ip_enabled:
            logger.info(f"Configuring static IP: {self.static_ip}")

            if not self.configure_static_ip_dhcpcd(interface):
                return False, "Failed to configure static IP"

            if not self.apply_network_configuration():
                return False, "Failed to apply network configuration"

            return True, f"Network configured with static IP {self.static_ip}"
        else:
            logger.info("Static IP disabled, using DHCP")
            current_ip = self.get_current_ip(interface)

            if current_ip:
                self.config["network"]["current_ip"] = current_ip
                self._save_config()
                return True, f"Using DHCP IP {current_ip}"
            else:
                return False, "Failed to get IP address"

    def get_network_status(self) -> Dict:
        """Get current network status."""
        interface = self.get_active_interface()
        current_ip = self.get_current_ip(interface)
        wifi_connected = self._is_wifi_connected()

        return {
            "active_interface": interface,
            "current_ip": current_ip,
            "static_ip_enabled": self.static_ip_enabled,
            "static_ip_configured": self.static_ip,
            "wifi_configured": self.config.get("network", {}).get("wifi_configured", False),
            "wifi_connected": wifi_connected,
            "wifi_ssid": self.config.get("network", {}).get("wifi_ssid"),
            "streaming_urls": {
                "mjpeg": f"http://{current_ip or self.static_ip}:5000/video_feed",
                "rtsp": f"rtsp://{current_ip or self.static_ip}:8554/ornimetrics/stream",
                "dashboard": f"http://{current_ip or self.static_ip}:5000/"
            }
        }


def main():
    """Test network service."""
    logging.basicConfig(level=logging.INFO)

    service = NetworkConfigService()

    # Get current status
    status = service.get_network_status()
    print("\nCurrent Network Status:")
    print(f"  Interface: {status['active_interface']}")
    print(f"  Current IP: {status['current_ip']}")
    print(f"  Static IP Enabled: {status['static_ip_enabled']}")
    print(f"  Static IP Target: {status['static_ip_configured']}")
    print(f"  WiFi Configured: {status['wifi_configured']}")
    print(f"  WiFi Connected: {status['wifi_connected']}")
    print(f"\nStreaming URLs:")
    print(f"  MJPEG: {status['streaming_urls']['mjpeg']}")
    print(f"  RTSP: {status['streaming_urls']['rtsp']}")
    print(f"  Dashboard: {status['streaming_urls']['dashboard']}")

    # Setup network if needed
    if status['current_ip'] != status['static_ip_configured']:
        print("\n[!] Current IP doesn't match target static IP")
        response = input("Configure static IP now? (y/n): ")

        if response.lower() == 'y':
            success, message = service.setup_network()
            if success:
                print(f"[✓] {message}")
            else:
                print(f"[✗] {message}")


if __name__ == "__main__":
    main()
