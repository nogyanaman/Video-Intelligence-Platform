import requests
import time
import sys
import os

# Configuration
API_URL = "http://localhost:8000"
# Change line 8 to this:
VIDEO_PATH = "test_video.mp4"
def upload_video():
    print(f"🚀 Uploading {VIDEO_PATH}...")
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: File {VIDEO_PATH} not found!")
        print("Please place a small .mp4 file in this folder and name it 'test_video.mp4'")
        sys.exit(1)

    with open(VIDEO_PATH, "rb") as f:
        response = requests.post(f"{API_URL}/videos", files={"file": f})
    
    if response.status_code == 202:
        data = response.json()
        print(f"✅ Upload successful! Video ID: {data['id']}")
        return data['id']
    else:
        print(f"❌ Upload failed: {response.text}")
        sys.exit(1)

def wait_for_processing(video_id):
    print("⏳ Waiting for processing pipeline...")
    while True:
        response = requests.get(f"{API_URL}/videos/{video_id}")
        data = response.json()
        status = data['status']
        
        # Get detailed job status
        jobs_response = requests.get(f"{API_URL}/videos/{video_id}/jobs")
        jobs = jobs_response.json()
        current_stage = next((j['stage'] for j in jobs if j['status'] == 'running'), "idle")
        
        sys.stdout.write(f"\r   Status: {status.upper()} | Current Stage: {current_stage}       ")
        sys.stdout.flush()

        if status == "ready":
            print("\n✨ Processing Complete!")
            return
        elif status == "failed":
            # Don't exit immediately! Check if it's actually dead or just retrying.
            # We will wait 10 more seconds to see if it switches back to "processing".
            print("\n⚠️ Status is FAILED. Waiting to see if it retries...")
            time.sleep(10)
            
            # Re-check status
            response = requests.get(f"{API_URL}/videos/{video_id}")
            new_status = response.json()['status']
            
            if new_status == "failed":
                print("❌ Processing definitely failed.")
                sys.exit(1)
            else:
                print("🔄 It switched back to processing! Continuing wait...")
        
        time.sleep(2)

def ask_question(video_id):
    query = "Summarize this." 
    print(f"\n🧠 Asking AI: '{query}'")
    
    response = requests.post(f"{API_URL}/queries", json={"query": query, "video_id": video_id})
    if response.status_code == 200:
        answer = response.json()
        print("-" * 50)
        print(f"🤖 Answer: {answer['answer']}")
        print(f"📊 Confidence: {answer['confidence']}")
        print("-" * 50)
    else:
        print(f"❌ Query failed: {response.text}")

if __name__ == "__main__":
    # Ensure requests is installed: pip install requests
    try:
        vid_id = upload_video()
        wait_for_processing(vid_id)
        ask_question(vid_id)
    except KeyboardInterrupt:
        print("\nTest cancelled.")