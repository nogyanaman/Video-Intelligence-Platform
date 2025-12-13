import requests
import time
import sys
import os

# ================= CONFIGURATION =================
API_URL = "http://localhost:8000"
VIDEO_FILENAME = "test_video.mp4"  # Make sure this file exists in the same folder!

# If your video is English, ask in English.
QUERY_TEXT = "any song used" 
# =================================================

def upload_video():
    """Uploads the video and returns the ID."""
    if not os.path.exists(VIDEO_FILENAME):
        print(f"\n❌ ERROR: File '{VIDEO_FILENAME}' not found!")
        print(f"👉 Please place a small .mp4 file in this folder and name it '{VIDEO_FILENAME}'")
        sys.exit(1)

    print(f"🚀 Uploading {VIDEO_FILENAME}...")
    try:
        with open(VIDEO_FILENAME, "rb") as f:
            response = requests.post(f"{API_URL}/videos", files={"file": f})
        
        if response.status_code == 202:
            data = response.json()
            video_id = data['id']
            print(f"✅ Upload successful! Video ID: {video_id}")
            return video_id
        else:
            print(f"❌ Upload failed: {response.text}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Is the API running at {API_URL}?")
        sys.exit(1)

def wait_for_processing(video_id):
    """Polls the API until the video status is 'ready'."""
    print("⏳ Waiting for processing pipeline...")
    start_time = time.time()
    
    while True:
        try:
            # Check main status
            response = requests.get(f"{API_URL}/videos/{video_id}")
            if response.status_code != 200:
                print(f"⚠️ API Error checking status: {response.status_code}")
                time.sleep(2)
                continue
                
            data = response.json()
            status = data['status']
            
            # Check specific job stages (for better progress bar)
            jobs_response = requests.get(f"{API_URL}/videos/{video_id}/jobs")
            current_stage = "initializing"
            if jobs_response.status_code == 200:
                jobs = jobs_response.json()
                # Find the first running job
                running_job = next((j for j in jobs if j['status'] == 'running'), None)
                if running_job:
                    current_stage = running_job['stage']
                elif status == 'processing':
                    current_stage = "queued"

            # Print status on the same line
            elapsed = int(time.time() - start_time)
            sys.stdout.write(f"\r   ⏱️ {elapsed}s | Status: {status.upper()} | Stage: {current_stage:<15} ")
            sys.stdout.flush()

            if status == "ready":
                print("\n\n✨ Processing Complete!")
                return
            
            elif status == "failed":
                print("\n\n⚠️ Status reported as FAILED. Double checking...")
                time.sleep(5) 
                # Final check to avoid false positives during retries
                final_check = requests.get(f"{API_URL}/videos/{video_id}").json()
                if final_check['status'] == "failed":
                    print("❌ Pipeline failed permanently.")
                    print("👉 Run 'docker compose logs -f worker' to see why.")
                    sys.exit(1)
                else:
                    print("🔄 It recovered! Continuing...")

            time.sleep(2)

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user.")
            sys.exit(0)

def ask_question(video_id):
    """Sends the query to the RAG endpoint."""
    print(f"\n🧠 Asking AI: '{QUERY_TEXT}'")
    
    payload = {
        "query": QUERY_TEXT,
        "video_id": video_id
    }
    
    try:
        response = requests.post(f"{API_URL}/queries", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("\n" + "="*50)
            if not data.get('sources'):
                print("⚠️  AI Response: The AI could not find relevant info.")
                print("   (Check if your query language matches the video language)")
            else:
                print(f"🤖 AI Answer: {data['answer']}")
                print(f"📊 Confidence: {data.get('confidence', 0)}")
            print("="*50 + "\n")
        else:
            print(f"❌ Query failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error asking question: {e}")

if __name__ == "__main__":
    print("--- 🎬 Video Intelligence Pipeline Test ---")
    vid_id = upload_video()
    wait_for_processing(vid_id)
    ask_question(vid_id)