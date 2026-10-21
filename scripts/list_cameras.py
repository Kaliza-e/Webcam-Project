"""
scripts/list_cameras.py
Utility script to probe, list, and test all connected cameras (internal webcams and external USB/IP cameras).
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.capture import list_available_cameras


def main():
    print("=" * 60)
    print(" CONNECTED CAMERA PROBE & DIAGNOSTICS")
    print("=" * 60)

    cameras = list_available_cameras(max_tested=5)

    if not cameras:
        print(" [!] No cameras detected on device indices 0 to 4.")
        print("     Ensure your external USB camera is plugged in.")
        return

    print(f" Found {len(cameras)} available camera(s):\n")

    for cam in cameras:
        idx = cam["index"]
        print(f" Camera Index [{idx}]:")
        print(f"   - Status: {'ONLINE (Ready to capture)' if cam['working'] else 'OPEN (Frame read failed)'}")
        print(f"   - Default Resolution: {cam['default_width']}x{cam['default_height']} @ {cam['fps']:.1f} FPS")

        # Test common HD resolutions
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        supported = []
        for rw, rh in [(1920, 1080), (1280, 720), (960, 540), (640, 480)]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (aw, ah) == (rw, rh) and (aw, ah) not in supported:
                supported.append((aw, ah))
        cap.release()

        if supported:
            res_str = ", ".join([f"{w}x{h}" for w, h in supported])
            print(f"   - Tested HD Modes: {res_str}")
        print("-" * 60)

    print("\nHow to use with Person Tracker:")
    print("  1. Launch with external camera:")
    print("     python -m src.main --camera 1")
    print("  2. Launch with external camera in 720p or 1080p:")
    print("     python -m src.main --camera 1 --width 1280 --height 720 --fullscreen")
    print("  3. Switch cameras live at any time:")
    print("     Press 'c' key in the live tracking window to swap between Camera 0 and 1.")
    print("=" * 60)


if __name__ == "__main__":
    main()
