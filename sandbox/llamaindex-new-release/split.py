import requests
import time

API_KEY = "llx-..."
BASE_URL = "https://api.cloud.llamaindex.ai/api/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# Define categories for splitting
categories = [
        {"name": "default"},
    ]

# Create split job
response = requests.post(
    f"{BASE_URL}/beta/split/jobs",
    headers=HEADERS,
    json={
        "document_input": {
            "type": "file_id",
            "value": "<your-file-id>",
        },
        "categories": categories,
        "splitting_strategy": {
            "allow_uncategorized": False,
        },
    },
)
response.raise_for_status()
job = response.json()

print(f"Split job created: {job['id']}")

# Poll for job completion
while job["status"] in ("pending", "processing"):
    time.sleep(2)
    response = requests.get(
        f"{BASE_URL}/beta/split/jobs/{job['id']}",
        headers=HEADERS,
    )
    response.raise_for_status()
    job = response.json()
    print(f"Status: {job['status']}")

# Check results
if job["status"] == "completed" and job.get("result"):
    segments = job["result"]["segments"]
    print(f"Split completed with {len(segments)} segments:")
    for segment in segments:
        print(f"  - {segment['category']}: Pages {segment['pages']} ({segment['confidence_category']} confidence)")
else:
    print(f"Job failed: {job.get('error_message')}")
  