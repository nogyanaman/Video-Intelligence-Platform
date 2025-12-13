import requests

# Your Video ID
VIDEO_ID = "4a92fb1e-d6c4-4f44-b07b-7f84d2d8ee43"
API_URL = "http://localhost:8000"

print(f"🕵️‍♀️ Inspecting Video: {VIDEO_ID}")

# 1. Get the Video Metadata (This usually contains the full transcript)
url = f"{API_URL}/videos/{VIDEO_ID}"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print("\n✅ Video Found!")
    print(f"Status: {data.get('status')}")
    
    # Check if transcript exists
    transcript = data.get('transcript', [])
    if transcript:
        print(f"\n📝 Transcript Found ({len(transcript)} segments):")
        print("-" * 40)
        # Print the first 3 lines of what the AI actually heard
        for seg in transcript[:5]: 
            print(f"[{seg['start']}s]: {seg['text']}")
        print("-" * 40)
        print("Copy one of the lines above exactly and try searching for it.")
    else:
        print("\n❌ Transcript is empty in the database.")
else:
    print(f"\n❌ API Error: {response.status_code} - {response.text}")