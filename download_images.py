from huggingface_hub import hf_hub_download
import os
os.makedirs("data/images", exist_ok=True)
for i in range(1, 201):  # Download 200 images
    try:
        hf_hub_download(repo_id="laion/laion-aesthetics", filename=f"image_{i}.jpg", local_dir="data/images")
        print(f"Downloaded image_{i}.jpg")
    except:
        print(f"Failed to download image_{i}.jpg")