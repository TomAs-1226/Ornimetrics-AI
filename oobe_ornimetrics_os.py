#!/usr/bin/env python3
"""
Ornimetrics OS - OOBE (Out Of Box Experience) Setup System

Bluetooth-first setup workflow for Ornimetrics OS:
1. Start Bluetooth service for mobile app pairing
2. Wait for account linking via app
3. Configure WiFi credentials via app
4. Set static IP for consistent streaming
5. Complete system configuration

This replaces keyboard/monitor setup with mobile app-based setup.
"""

import os
import sys
import json
import subprocess
import logging
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Setup configuration
SETUP_MARKER_FILE = ".oobe_completed"
CONFIG_FILE = "ornimetrics_os_config.json"
SCRIPT_DIR = Path(__file__).resolve().parent


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_banner():
    """Print Ornimetrics OS welcome banner."""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 70)
    print("   🐦 ORNIMETRICS OS - FIRST TIME SETUP")
    print("=" * 70)
    print(f"{Colors.RESET}")
    print(f"{Colors.BLUE}Welcome to Ornimetrics OS!{Colors.RESET}\n")


def check_setup_needed() -> bool:
    """Check if OOBE setup has already been completed."""
    marker_path = SCRIPT_DIR / SETUP_MARKER_FILE

    if marker_path.exists():
        try:
            with open(marker_path, 'r') as f:
                setup_info = json.load(f)

            completed_date = setup_info.get("completed_date")
            version = setup_info.get("version", "unknown")

            logger.info(f"Setup already completed on {completed_date} (version {version})")
            print(f"{Colors.GREEN}✅ Ornimetrics OS already configured{Colors.RESET}")
            print(f"   Setup completed: {completed_date}")
            print(f"   Version: {version}")
            print(f"\n💡 To re-run setup, delete: {marker_path}\n")
            return False
        except Exception as e:
            logger.warning(f"Invalid setup marker file: {e}")
            return True

    return True


def load_config() -> Dict:
    """Load Ornimetrics OS configuration."""
    config_path = SCRIPT_DIR / CONFIG_FILE

    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)

    # Return default config
    logger.warning("Config not found, using defaults")
    return {}


def save_config(config: Dict):
    """Save Ornimetrics OS configuration."""
    config_path = SCRIPT_DIR / CONFIG_FILE
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def initialize_config() -> Dict:
    """Initialize default Ornimetrics OS configuration."""
    print(f"\n{Colors.BOLD}1. Initializing Configuration{Colors.RESET}")
    print("=" * 50)

    config_path = SCRIPT_DIR / CONFIG_FILE

    if config_path.exists():
        print(f"{Colors.BLUE}ℹ️{Colors.RESET}  Configuration exists, loading...")
        config = load_config()
    else:
        print(f"{Colors.CYAN}Creating default configuration...{Colors.RESET}")

        # Generate unique device ID
        device_id = str(uuid.uuid4())

        # Get hostname
        import socket
        hostname = socket.gethostname()

        config = {
            "_branding": {
                "product_name": "Ornimetrics OS",
                "version": "1.0.0",
                "build_date": datetime.now().isoformat()
            },
            "system": {
                "device_id": device_id,
                "device_name": hostname,
                "first_boot": datetime.now().isoformat()
            },
            "account": {
                "linked": False,
                "user_id": None,
                "account_email": None,
                "account_token": None,
                "linked_timestamp": None,
                "feeder_name": "My Feeder",
                "sharing_enabled": False
            },
            "network": {
                "wifi_configured": False,
                "wifi_ssid": None,
                "static_ip_enabled": True,
                "static_ip": "192.168.1.200",
                "subnet_mask": "255.255.255.0",
                "gateway": "192.168.1.1",
                "dns_servers": ["8.8.8.8", "8.8.4.4"],
                "current_ip": None,
                "configured": False
            },
            "bluetooth": {
                "enabled": True,
                "device_name": f"Ornimetrics-{hostname}",
                "allow_setup_via_bluetooth": True,
                "pin_required": False,
                "pin_code": "1234",
                "paired_devices": []
            },
            "streaming": {
                "enabled": True,
                "mjpeg_enabled": True,
                "mjpeg_port": 5000,
                "rtsp_enabled": True,
                "rtsp_port": 8554,
                "stream_quality": "high"
            },
            "mobile_app": {
                "pairing_required": True,
                "allowed_operations": [
                    "view_stream",
                    "configure_settings",
                    "view_detections",
                    "link_account",
                    "configure_wifi"
                ]
            },
            "firebase": {
                "structure": "user_based",
                "path_template": "/users/{user_id}/feeders/{device_id}/",
                "data_publishing_enabled": True,
                "user_data_only": True,
                "no_public_sharing": True
            },
            "features": {
                "bird_detection": {"enabled": True},
                "individual_recognition": {"enabled": True},
                "auto_training": {"enabled": True},
                "3d_camera": {"enabled": True, "fallback_to_2d": True}
            }
        }

        save_config(config)
        print(f"{Colors.GREEN}✅{Colors.RESET} Configuration created")
        print(f"   Device ID: {device_id}")
        print(f"   Device Name: {hostname}")

    return config


def start_bluetooth_service() -> Optional[object]:
    """Start Bluetooth service for mobile app pairing."""
    print(f"\n{Colors.BOLD}2. Starting Bluetooth Service{Colors.RESET}")
    print("=" * 50)

    try:
        from src.bluetooth_service import BluetoothSetupService, configure_wifi_callback

        service = BluetoothSetupService(CONFIG_FILE)

        # Register WiFi configuration callback
        service.register_callback('wifi_configured', configure_wifi_callback)

        if service.start():
            print(f"{Colors.GREEN}✅{Colors.RESET} Bluetooth service started")

            pairing_info = service.get_pairing_info()
            print(f"\n{Colors.CYAN}📱 Pairing Information:{Colors.RESET}")
            print(f"   Device Name: {pairing_info['device_name']}")
            print(f"   Service: {pairing_info['service_name']}")

            if pairing_info['pin_required']:
                print(f"   PIN Code: {pairing_info['pin_code']}")

            return service
        else:
            print(f"{Colors.RED}❌{Colors.RESET} Failed to start Bluetooth service")
            return None

    except Exception as e:
        logger.error(f"Bluetooth service error: {e}")
        print(f"{Colors.RED}❌{Colors.RESET} Bluetooth not available: {e}")
        return None


def wait_for_account_linking(service: object, timeout: int = 600) -> bool:
    """Wait for account linking via mobile app.

    Args:
        service: BluetoothSetupService instance
        timeout: Timeout in seconds (default 10 minutes)

    Returns:
        True if account linked successfully
    """
    print(f"\n{Colors.BOLD}3. Waiting for Mobile App Pairing{Colors.RESET}")
    print("=" * 50)

    print(f"\n{Colors.CYAN}📱 Open the Ornimetrics mobile app and follow these steps:{Colors.RESET}\n")
    print(f"   1. Tap 'Add New Feeder'")
    print(f"   2. Select this device from Bluetooth list")
    print(f"   3. Complete account linking")
    print(f"   4. Configure WiFi credentials\n")

    print(f"{Colors.YELLOW}⏳ Waiting for pairing... (timeout: {timeout//60} minutes){Colors.RESET}\n")

    start_time = time.time()

    while time.time() - start_time < timeout:
        # Check if account has been linked
        config = load_config()

        if config.get("account", {}).get("linked"):
            print(f"\n{Colors.GREEN}✅ Account linked successfully!{Colors.RESET}")
            print(f"   User: {config['account'].get('account_email')}")
            print(f"   Feeder Name: {config['account'].get('feeder_name')}")
            return True

        # Show progress dots
        elapsed = int(time.time() - start_time)
        if elapsed % 10 == 0:
            print(f"   Still waiting... ({elapsed}s elapsed)", end='\r')

        time.sleep(1)

    print(f"\n{Colors.RED}❌ Timeout waiting for account linking{Colors.RESET}")
    return False


def configure_network() -> bool:
    """Configure network with static IP."""
    print(f"\n{Colors.BOLD}4. Configuring Network{Colors.RESET}")
    print("=" * 50)

    try:
        from src.network_service import NetworkConfigService

        network = NetworkConfigService(CONFIG_FILE)

        # Check if WiFi is configured
        config = load_config()
        wifi_configured = config.get("network", {}).get("wifi_configured", False)

        if wifi_configured:
            print(f"{Colors.GREEN}✅{Colors.RESET} WiFi configured via app")
            print(f"   SSID: {config['network'].get('wifi_ssid')}")
        else:
            print(f"{Colors.YELLOW}⚠️{Colors.RESET}  WiFi not configured (using Ethernet)")

        # Setup network (apply static IP)
        print(f"\n{Colors.CYAN}Configuring static IP...{Colors.RESET}")
        success, message = network.setup_network()

        if success:
            print(f"{Colors.GREEN}✅{Colors.RESET} {message}")

            # Show network status
            status = network.get_network_status()
            print(f"\n{Colors.CYAN}Network Status:{Colors.RESET}")
            print(f"   Interface: {status['active_interface']}")
            print(f"   IP Address: {status['current_ip']}")
            print(f"\n{Colors.CYAN}Streaming URLs:{Colors.RESET}")
            print(f"   Dashboard: {status['streaming_urls']['dashboard']}")
            print(f"   MJPEG: {status['streaming_urls']['mjpeg']}")
            print(f"   RTSP: {status['streaming_urls']['rtsp']}")

            return True
        else:
            print(f"{Colors.RED}❌{Colors.RESET} {message}")
            return False

    except Exception as e:
        logger.error(f"Network configuration error: {e}")
        print(f"{Colors.RED}❌{Colors.RESET} Network configuration failed: {e}")
        return False


def install_dependencies() -> bool:
    """Install required Python dependencies."""
    print(f"\n{Colors.BOLD}5. Installing Dependencies{Colors.RESET}")
    print("=" * 50)

    # Check if requirements file exists
    requirements_file = SCRIPT_DIR / "requirements.txt"

    if not requirements_file.exists():
        print(f"{Colors.YELLOW}⚠️{Colors.RESET}  requirements.txt not found, skipping")
        return True

    try:
        print(f"{Colors.CYAN}Installing packages from requirements.txt...{Colors.RESET}")

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        print(f"{Colors.GREEN}✅{Colors.RESET} Dependencies installed")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        print(f"{Colors.RED}❌{Colors.RESET} Dependency installation failed")
        return False


def create_directories() -> bool:
    """Create necessary directories."""
    print(f"\n{Colors.BOLD}6. Creating Directory Structure{Colors.RESET}")
    print("=" * 50)

    directories = [
        "data",
        "data/species_prototypes",
        "data/training_samples",
        "logs",
        "models",
        "models/checkpoints",
        "models/test_results",
    ]

    for dir_name in directories:
        dir_path = SCRIPT_DIR / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"{Colors.GREEN}✅{Colors.RESET} Created: {dir_name}")
        else:
            print(f"{Colors.BLUE}ℹ️{Colors.RESET}  Exists: {dir_name}")

    return True


def generate_species_prototypes() -> bool:
    """Generate initial species prototypes."""
    print(f"\n{Colors.BOLD}7. Generating Species Prototypes{Colors.RESET}")
    print("=" * 50)

    config_path = SCRIPT_DIR / "species_3d_support.json"
    proto_dir = SCRIPT_DIR / "data" / "species_prototypes"

    if not config_path.exists():
        print(f"{Colors.YELLOW}⚠️{Colors.RESET}  species_3d_support.json not found, skipping")
        return True

    # Check if prototypes already exist
    existing_plys = list(proto_dir.glob("*.ply"))

    if existing_plys:
        print(f"{Colors.BLUE}ℹ️{Colors.RESET}  Found {len(existing_plys)} existing prototypes, skipping generation")
        return True

    try:
        from src.generate_species_prototypes import generate_species_prototypes

        print(f"{Colors.CYAN}Generating synthetic point clouds...{Colors.RESET}")

        generated = generate_species_prototypes(
            config_path=str(config_path),
            output_dir=str(proto_dir)
        )

        if generated:
            print(f"{Colors.GREEN}✅{Colors.RESET} Generated prototypes for {len(generated)} species")
            return True
        else:
            print(f"{Colors.YELLOW}⚠️{Colors.RESET}  No prototypes generated (non-critical)")
            return True

    except Exception as e:
        logger.error(f"Prototype generation error: {e}")
        print(f"{Colors.YELLOW}⚠️{Colors.RESET}  Prototype generation failed (non-critical)")
        return True


def save_setup_marker(config: Dict) -> bool:
    """Save setup completion marker."""
    marker_path = SCRIPT_DIR / SETUP_MARKER_FILE

    marker_data = {
        "completed": True,
        "completed_date": datetime.now().isoformat(),
        "version": config.get("_branding", {}).get("version", "1.0.0"),
        "device_id": config.get("system", {}).get("device_id"),
        "account_linked": config.get("account", {}).get("linked", False),
        "setup_method": "bluetooth_app"
    }

    try:
        with open(marker_path, 'w') as f:
            json.dump(marker_data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save setup marker: {e}")
        return False


def print_completion_message(config: Dict):
    """Print setup completion message."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}")
    print("=" * 70)
    print("   🎉 ORNIMETRICS OS SETUP COMPLETE!")
    print("=" * 70)
    print(f"{Colors.RESET}")

    print(f"\n{Colors.GREEN}Your Ornimetrics OS feeder is ready!{Colors.RESET}\n")

    # Show device info
    print(f"{Colors.BOLD}Device Information:{Colors.RESET}")
    print(f"   Feeder Name: {config['account'].get('feeder_name')}")
    print(f"   Device ID: {config['system'].get('device_id')}")
    print(f"   Account: {config['account'].get('account_email')}\n")

    # Show access info
    current_ip = config.get("network", {}).get("current_ip", "192.168.1.200")
    print(f"{Colors.BOLD}Access Your Feeder:{Colors.RESET}")
    print(f"   Dashboard: http://{current_ip}:5000/")
    print(f"   Mobile App: Already paired")
    print(f"   MJPEG Stream: http://{current_ip}:5000/video_feed")
    print(f"   RTSP Stream: rtsp://{current_ip}:8554/ornimetrics/stream\n")

    # Next steps
    print(f"{Colors.BOLD}Next Steps:{Colors.RESET}")
    print(f"   1. System will start automatically")
    print(f"   2. Visit dashboard to monitor detections")
    print(f"   3. Use mobile app for full control\n")

    print(f"{Colors.GREEN}Happy bird watching! 🐦{Colors.RESET}\n")


def main():
    """Main OOBE setup routine for Ornimetrics OS."""
    os.chdir(SCRIPT_DIR)

    print_banner()

    # Check if setup already done
    if not check_setup_needed():
        return 0

    # Step 1: Initialize configuration
    try:
        config = initialize_config()
    except Exception as e:
        logger.error(f"Failed to initialize config: {e}")
        print(f"{Colors.RED}❌ Configuration initialization failed{Colors.RESET}")
        return 1

    # Step 2: Start Bluetooth service
    bluetooth_service = start_bluetooth_service()

    if not bluetooth_service:
        print(f"\n{Colors.YELLOW}⚠️  Bluetooth not available{Colors.RESET}")
        print(f"{Colors.YELLOW}   Manual setup required via web interface{Colors.RESET}\n")
        # Don't fail - allow manual setup
    else:
        # Step 3: Wait for account linking
        if not wait_for_account_linking(bluetooth_service, timeout=600):
            print(f"\n{Colors.YELLOW}⚠️  Account linking timeout{Colors.RESET}")
            print(f"{Colors.YELLOW}   You can complete setup later via web interface{Colors.RESET}\n")
            # Don't fail - allow manual completion

        # Stop Bluetooth service (account linked, WiFi configured)
        bluetooth_service.stop()

    # Step 4: Configure network
    if not configure_network():
        print(f"{Colors.YELLOW}⚠️  Network configuration incomplete (non-critical){Colors.RESET}")

    # Step 5: Install dependencies
    if not install_dependencies():
        print(f"{Colors.YELLOW}⚠️  Dependency installation incomplete (non-critical){Colors.RESET}")

    # Step 6: Create directories
    if not create_directories():
        print(f"{Colors.RED}❌ Failed to create directories{Colors.RESET}")
        return 1

    # Step 7: Generate prototypes
    generate_species_prototypes()

    # Save completion marker
    config = load_config()  # Reload to get updates

    if not save_setup_marker(config):
        print(f"{Colors.YELLOW}⚠️  Warning: Could not save setup marker{Colors.RESET}")

    # Show completion message
    print_completion_message(config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
