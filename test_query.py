import requests

# Keep your existing Video ID
VIDEO_ID = "..." 
API_URL = "http://localhost:8000"

# --- CHANGE THIS QUERY ---
# Old: "What happens in this video?"
# New (Hindi): "इस शख्स ने क्या चुराया है?" (What did this person steal?)
# Or try specific keywords: "Google aur Instagram par kya hai?"
QUERY = "Summarize this." 

print(f"🧠 Asking AI about video {VIDEO_ID}...")

try:
    response = requests.post(
        f"{API_URL}/queries",
        json={"query": QUERY, "video_id": VIDEO_ID}
    )
# ... rest of the script stays the same ...

    if response.status_code == 200:
        data = response.json()
        print("\n" + "="*50)
        print(f"🤖 AI Answer: {data['answer']}")
        print(f"📊 Confidence: {data['confidence']}")
        print("="*50)
        print("\nSources used:")
        for source in data['sources']:
            print(f"- [{source['start_ms']}ms]: {source['text']}")
    else:
        print(f"❌ Query failed: {response.text}")

except Exception as e:
    print(f"❌ Connection error: {e}")