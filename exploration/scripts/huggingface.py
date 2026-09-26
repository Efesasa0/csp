import os
from pathlib import Path
from huggingface_hub import snapshot_download

def fetch_and_explore_dataset(
    save_path: str = str(Path(__file__).resolve().parents[1] / "data"),
):
    # The Hugging Face identifier
    dataset_name = "AST-Revisited/contrastive-stubs"
    print(f"Downloading all files from repository: {dataset_name} to {save_path}...")
    
    try:
        # Ensure the directory exists
        os.makedirs(save_path, exist_ok=True)
        
        # Download the entire repository directly to the specified folder
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=save_path,
        )
        print("All files successfully downloaded!")
            
    except Exception as e:
        print(f"An error occurred while fetching the dataset: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=None, help="Directory to download files into")
    args = parser.parse_args()
    kwargs = {"save_path": args.out_dir} if args.out_dir else {}
    fetch_and_explore_dataset(**kwargs)
