#!/usr/bin/env python3
"""
OOBE (Out Of Box Experience) Setup System

Automatically configures the Ornimetrics bird detection system on first run.
Checks dependencies, installs missing components, generates initial data,
and configures system for operation.

This script is designed to be idempotent and only performs setup once.
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import platform

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Setup configuration
SETUP_MARKER_FILE = ".oobe_completed"
FEEDER_CONFIG_FILE = "feeder_config.json"
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
    """Print welcome banner."""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 70)
    print("   🐦 ORNIMETRICS BIRD DETECTION SYSTEM - FIRST TIME SETUP")
    print("=" * 70)
    print(f"{Colors.RESET}")
    print(f"{Colors.BLUE}Initializing system configuration...{Colors.RESET}\n")


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
            print(f"{Colors.GREEN}✅ System already configured{Colors.RESET}")
            print(f"   Setup completed: {completed_date}")
            print(f"   Version: {version}")
            print(f"\n💡 To re-run setup, delete: {marker_path}\n")
            return False
        except Exception as e:
            logger.warning(f"Invalid setup marker file: {e}")
            # Continue with setup if marker is corrupted
            return True

    return True


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is adequate."""
    version_info = sys.version_info
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 7):
        return False, f"Python {version_info.major}.{version_info.minor} (3.7+ required)"
    return True, f"Python {version_info.major}.{version_info.minor}.{version_info.micro}"


def check_system_info() -> Dict:
    """Gather system information."""
    return {
        "platform": platform.system(),
        "platform_version": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "is_raspberry_pi": Path("/proc/device-tree/model").exists() and "Raspberry Pi" in open("/proc/device-tree/model").read() if Path("/proc/device-tree/model").exists() else False
    }


def check_package(package_name: str, import_name: Optional[str] = None) -> bool:
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name.replace("-", "_")

    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def install_package(package_name: str) -> bool:
    """Install a Python package using pip."""
    try:
        logger.info(f"Installing {package_name}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        logger.info(f"✅ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install {package_name}: {e}")
        return False


def check_and_install_dependencies() -> Tuple[bool, List[str]]:
    """Check and install required dependencies."""
    print(f"\n{Colors.BOLD}1. Checking Dependencies{Colors.RESET}")
    print("=" * 50)

    # Core dependencies
    core_deps = {
        "numpy": "numpy",
        "opencv-python": "cv2",
        "torch": "torch",
        "torchvision": "torchvision",
        "ultralytics": "ultralytics",
        "flask": "flask",
        "flask-cors": "flask_cors",
    }

    # Optional dependencies (for 3D mode)
    optional_deps = {
        "open3d": "open3d",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
    }

    # Firebase dependencies (optional)
    firebase_deps = {
        "firebase-admin": "firebase_admin",
    }

    missing_core = []
    missing_optional = []
    missing_firebase = []

    # Check core dependencies
    for package, import_name in core_deps.items():
        if check_package(package, import_name):
            print(f"{Colors.GREEN}✅{Colors.RESET} {package}")
        else:
            print(f"{Colors.RED}❌{Colors.RESET} {package} (missing)")
            missing_core.append(package)

    # Check optional dependencies
    for package, import_name in optional_deps.items():
        if check_package(package, import_name):
            print(f"{Colors.GREEN}✅{Colors.RESET} {package} (optional)")
        else:
            print(f"{Colors.YELLOW}⚠️{Colors.RESET}  {package} (optional, missing)")
            missing_optional.append(package)

    # Check Firebase
    for package, import_name in firebase_deps.items():
        if check_package(package, import_name):
            print(f"{Colors.GREEN}✅{Colors.RESET} {package} (optional)")
        else:
            print(f"{Colors.YELLOW}⚠️{Colors.RESET}  {package} (optional, missing)")
            missing_firebase.append(package)

    # Install missing packages
    all_missing = missing_core + missing_optional + missing_firebase
    if not all_missing:
        print(f"\n{Colors.GREEN}✅ All dependencies satisfied{Colors.RESET}")
        return True, []

    print(f"\n{Colors.YELLOW}Installing missing packages...{Colors.RESET}")

    failed = []

    # Install core (required)
    for package in missing_core:
        if not install_package(package):
            failed.append(package)

    # Install optional (best effort)
    for package in missing_optional:
        install_package(package)  # Don't fail on optional deps

    # Install Firebase (best effort)
    for package in missing_firebase:
        install_package(package)

    if failed:
        print(f"\n{Colors.RED}❌ Failed to install required packages: {', '.join(failed)}{Colors.RESET}")
        return False, failed

    print(f"\n{Colors.GREEN}✅ All required dependencies installed{Colors.RESET}")
    return True, []


def create_directories() -> bool:
    """Create necessary directories."""
    print(f"\n{Colors.BOLD}2. Creating Directory Structure{Colors.RESET}")
    print("=" * 50)

    directories = [
        "data",
        "data/species_prototypes",
        "logs",
        "backups",
        "models",
    ]

    for dir_name in directories:
        dir_path = SCRIPT_DIR / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"{Colors.GREEN}✅{Colors.RESET} Created: {dir_name}")
        else:
            print(f"{Colors.BLUE}ℹ️{Colors.RESET}  Exists: {dir_name}")

    return True


def generate_initial_prototypes() -> bool:
    """Generate initial species prototypes."""
    print(f"\n{Colors.BOLD}3. Generating Species Prototypes{Colors.RESET}")
    print("=" * 50)

    # Check if config exists
    config_path = SCRIPT_DIR / "species_3d_support.json"
    if not config_path.exists():
        print(f"{Colors.YELLOW}⚠️  Species configuration not found, skipping prototypes{Colors.RESET}")
        return True

    # Check if prototypes already exist
    proto_dir = SCRIPT_DIR / "data" / "species_prototypes"
    existing_plys = list(proto_dir.glob("*.ply"))

    if existing_plys:
        print(f"{Colors.BLUE}ℹ️{Colors.RESET}  Found {len(existing_plys)} existing prototypes")
        response = input("   Regenerate prototypes? (y/N): ").strip().lower()
        if response != 'y':
            print(f"{Colors.GREEN}✅{Colors.RESET} Keeping existing prototypes")
            return True

    # Generate prototypes
    try:
        from src.generate_species_prototypes import generate_species_prototypes
        print(f"{Colors.CYAN}Generating synthetic point clouds...{Colors.RESET}")

        generated = generate_species_prototypes(
            config_path=str(config_path),
            output_dir=str(proto_dir)
        )

        if generated:
            print(f"\n{Colors.GREEN}✅ Generated prototypes for {len(generated)} species{Colors.RESET}")
            return True
        else:
            print(f"{Colors.YELLOW}⚠️  No prototypes generated (check configuration){Colors.RESET}")
            return True  # Don't fail setup

    except Exception as e:
        logger.error(f"Failed to generate prototypes: {e}")
        print(f"{Colors.YELLOW}⚠️  Prototype generation failed (non-critical){Colors.RESET}")
        return True  # Don't fail setup for this


def create_feeder_config() -> bool:
    """Create feeder configuration for future app connectivity."""
    print(f"\n{Colors.BOLD}4. Creating Feeder Configuration{Colors.RESET}")
    print("=" * 50)

    config_path = SCRIPT_DIR / FEEDER_CONFIG_FILE

    if config_path.exists():
        print(f"{Colors.BLUE}ℹ️{Colors.RESET}  Configuration already exists")
        return True

    # Generate unique feeder ID
    import uuid
    feeder_id = str(uuid.uuid4())

    # Get system hostname
    import socket
    hostname = socket.gethostname()

    config = {
        "feeder": {
            "id": feeder_id,
            "name": hostname,
            "location": {
                "description": "Set your location",
                "latitude": None,
                "longitude": None,
                "timezone": "America/New_York"
            },
            "hardware": {
                "has_3d_camera": False,  # Will be auto-detected
                "has_hailo": False,      # Will be auto-detected
                "camera_model": "auto-detect",
                "servo_channel": 1
            },
            "status": {
                "enabled": True,
                "last_boot": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        },
        "app_connectivity": {
            "_comment": "Configuration for future mobile app integration",
            "enabled": False,
            "api_endpoint": "http://localhost:5000",
            "authentication": {
                "enabled": False,
                "api_key": None,
                "username": None
            },
            "sync": {
                "enabled": False,
                "cloud_endpoint": None,
                "sync_interval_minutes": 60,
                "sync_images": True,
                "sync_individuals": True,
                "sync_statistics": True
            }
        },
        "multi_feeder_network": {
            "_comment": "Support for multiple feeders (future feature)",
            "enabled": False,
            "network_id": None,
            "primary_feeder": True,
            "connected_feeders": [],
            "share_individual_database": False
        },
        "monitoring": {
            "enable_web_dashboard": True,
            "enable_rest_api": True,
            "enable_firebase": False,
            "log_level": "INFO"
        }
    }

    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"{Colors.GREEN}✅{Colors.RESET} Created feeder configuration")
        print(f"   Feeder ID: {feeder_id}")
        print(f"   Name: {hostname}")
        print(f"\n   {Colors.CYAN}💡 Edit {FEEDER_CONFIG_FILE} to customize settings{Colors.RESET}")

        return True

    except Exception as e:
        logger.error(f"Failed to create feeder config: {e}")
        return False


def check_hardware() -> Dict[str, bool]:
    """Detect available hardware."""
    print(f"\n{Colors.BOLD}5. Detecting Hardware{Colors.RESET}")
    print("=" * 50)

    hardware = {
        "hailo": False,
        "cs20_depth_camera": False,
        "usb_camera": False
    }

    # Check for Hailo
    try:
        import importlib.util
        if importlib.util.find_spec("hailo_platform") is not None:
            hardware["hailo"] = True
            print(f"{Colors.GREEN}✅{Colors.RESET} Hailo-8 AI Hat detected")
        else:
            print(f"{Colors.YELLOW}⚠️{Colors.RESET}  Hailo-8 AI Hat not found")
    except:
        print(f"{Colors.YELLOW}⚠️{Colors.RESET}  Hailo-8 AI Hat not found")

    # Check for cameras
    try:
        import cv2
        # Try to open camera
        for idx in range(3):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                hardware["usb_camera"] = True
                cap.release()
                print(f"{Colors.GREEN}✅{Colors.RESET} USB camera detected (device {idx})")
                break
        else:
            print(f"{Colors.YELLOW}⚠️{Colors.RESET}  No USB camera detected")
    except Exception as e:
        print(f"{Colors.YELLOW}⚠️{Colors.RESET}  Could not check for cameras: {e}")

    # Check for CS20 (look for specific video device properties)
    # This is a placeholder - actual detection would need CS20-specific checks
    print(f"{Colors.BLUE}ℹ️{Colors.RESET}  CS20 depth camera will be auto-detected at runtime")

    return hardware


def save_setup_marker(system_info: Dict, hardware: Dict) -> bool:
    """Save setup completion marker."""
    marker_path = SCRIPT_DIR / SETUP_MARKER_FILE

    marker_data = {
        "completed": True,
        "completed_date": datetime.now().isoformat(),
        "version": "1.0.0",
        "system_info": system_info,
        "detected_hardware": hardware,
        "python_version": sys.version,
    }

    try:
        with open(marker_path, 'w') as f:
            json.dump(marker_data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save setup marker: {e}")
        return False


def print_next_steps():
    """Print next steps for user."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}")
    print("=" * 70)
    print("   🎉 SETUP COMPLETE!")
    print("=" * 70)
    print(f"{Colors.RESET}")

    print(f"\n{Colors.GREEN}Next Steps:{Colors.RESET}\n")

    print(f"{Colors.BOLD}1. Configure your system:{Colors.RESET}")
    print(f"   Edit: config_3d_detection.json")
    print(f"   Edit: feeder_config.json")
    print(f"   Edit: trap_settings_full.json\n")

    print(f"{Colors.BOLD}2. Start the detection system:{Colors.RESET}")
    print(f"   Manual: ./start_detection_system.sh")
    print(f"   Auto-start: sudo ./install_service.sh\n")

    print(f"{Colors.BOLD}3. Access the web dashboard:{Colors.RESET}")
    print(f"   http://your-raspberry-pi-ip:5000/\n")

    print(f"{Colors.BOLD}4. View logs:{Colors.RESET}")
    print(f"   Service: sudo journalctl -u ornimetrics-detection -f")
    print(f"   Manual: Check console output\n")

    print(f"{Colors.CYAN}📖 Documentation:{Colors.RESET}")
    print(f"   - QUICKSTART_3D.md - Quick start guide")
    print(f"   - BIRD_3D_RECOGNITION.md - Full documentation")
    print(f"   - WEB_SERVER_GUIDE.md - Web server guide\n")

    print(f"{Colors.GREEN}Happy bird watching! 🐦{Colors.RESET}\n")


def main():
    """Main OOBE setup routine."""
    os.chdir(SCRIPT_DIR)

    print_banner()

    # Check if setup already done
    if not check_setup_needed():
        return 0

    # Check Python version
    python_ok, python_version = check_python_version()
    if not python_ok:
        print(f"{Colors.RED}❌ Incompatible Python version: {python_version}{Colors.RESET}")
        return 1

    print(f"{Colors.GREEN}✅ Python version: {python_version}{Colors.RESET}")

    # Get system info
    system_info = check_system_info()
    print(f"{Colors.GREEN}✅ System: {system_info['platform']} {system_info['machine']}{Colors.RESET}")

    if system_info["is_raspberry_pi"]:
        print(f"{Colors.GREEN}✅ Running on Raspberry Pi{Colors.RESET}")

    # Run setup steps
    steps = [
        ("Dependencies", check_and_install_dependencies),
        ("Directories", create_directories),
        ("Prototypes", generate_initial_prototypes),
        ("Feeder Config", create_feeder_config),
    ]

    for step_name, step_func in steps:
        try:
            result = step_func()
            if isinstance(result, tuple):
                success, _ = result
            else:
                success = result

            if not success:
                print(f"\n{Colors.RED}❌ Setup failed at step: {step_name}{Colors.RESET}")
                return 1
        except Exception as e:
            logger.error(f"Error in step '{step_name}': {e}", exc_info=True)
            print(f"\n{Colors.RED}❌ Setup failed at step: {step_name}{Colors.RESET}")
            print(f"   Error: {e}")
            return 1

    # Detect hardware (informational only)
    hardware = check_hardware()

    # Save completion marker
    if not save_setup_marker(system_info, hardware):
        print(f"\n{Colors.YELLOW}⚠️  Warning: Could not save setup marker{Colors.RESET}")

    # Show next steps
    print_next_steps()

    return 0


if __name__ == "__main__":
    sys.exit(main())
