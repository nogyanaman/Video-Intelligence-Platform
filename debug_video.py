import requests
import json

# Your Video ID
VIDEO_ID = "..."
API_URL = "http://localhost:8000"

print(f"🕵️‍♂️ Inspecting Video: {VIDEO_ID}")

# 1. Get Transcript
print("\n--- 🗣️ TRANSCRIPT ---")
try:
    resp = requests.get(f"{API_URL}/videos/{VIDEO_ID}/transcript")
    if resp.status_code == 200:
        data = resp.json()
        segments = data.get("segments", [])
        if not segments:
            print("⚠️ NO SPEECH DETECTED (Video might be silent or music-only)")
        for seg in segments:
            print(f"[{seg['start_ms']}ms]: {seg['text']}")
    else:
        print(f"Error: {resp.text}")
except Exception as e:
    print(f"Connection error: {e}")

# 2. Get Visual Objects
print("\n--- 👁️ VISUAL OBJECTS (YOLO) ---")
try:
    resp = requests.get(f"{API_URL}/videos/{VIDEO_ID}/detections")
    if resp.status_code == 200:
        data = resp.json()
        objects = data.get("unique_objects", {})
        if not objects:
            print("⚠️ NO OBJECTS DETECTED")
        for obj, count in objects.items():
            print(f"- {obj}: detected {count} times")
    else:
        print(f"Error: {resp.text}")
except Exception as e:
    print(f"Connection error: {e}")