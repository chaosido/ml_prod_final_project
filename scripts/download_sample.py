import os
import soundfile as sf
from datasets import load_dataset, Audio

# Dataset configuration
DATASET_NAME = "facebook/voxpopuli"
CONFIG_NAME = "nl"  # Dutch
SPLIT = "train"
NUM_SAMPLES = 5

# Determine output directory based on environment
# In Docker, we map ./data to /opt/data
if os.path.exists("/opt/data"):
    OUTPUT_DIR = "/opt/data/incoming"
else:
    # Local fallback (assuming running from project root)
    OUTPUT_DIR = os.path.join(os.getcwd(), "data", "incoming")

def download_sample():
    print(f"Loading dataset {DATASET_NAME} ({CONFIG_NAME})...")
    
    # Load dataset in streaming mode to avoid downloading everything
    # trust_remote_code=True is often needed for some HF datasets, fleurs usually implies it
    ds = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT, streaming=True, trust_remote_code=True)
    
    # Ensure directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving samples to {OUTPUT_DIR}...")
    
    # Iterate and save
    count = 0
    for i, item in enumerate(ds):
        if count >= NUM_SAMPLES:
            break
            
        audio = item["audio"]
        # item['audio'] contains: {'path': ..., 'array': ..., 'sampling_rate': ...}
        
        # Original filename might be a path or None in streaming
        audio_id = item.get("id", f"sample_{i}")
        filename = f"{audio_id}.wav"
        out_path = os.path.join(OUTPUT_DIR, filename)
        
        
        # Save audio
        sf.write(out_path, audio["array"], audio["sampling_rate"])
        
        # Save transcript for reference
        txt_path = out_path.replace(".wav", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(item.get("raw_text", "") or item.get("normalized_text", ""))
            
        print(f"Saved {out_path} and transcript")
        count += 1
        
    print(f"Successfully downloaded {count} samples.")

if __name__ == "__main__":
    # Ensure dependencies are installed (basic check)
    try:
        import datasets
        import soundfile
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please run: pip install datasets soundfile")
        exit(1)
        
    download_sample()
