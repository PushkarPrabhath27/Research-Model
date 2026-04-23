import argparse
import os
import sys
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--username", default="Pushkar27")
    args = parser.parse_args()
    
    api = HfApi(token=args.token)
    print(f"Authenticated as: {api.whoami()['name']}")
    
    repos = [
        ("GriceBench-Detector", "best_model_v2.pt"),
        ("GriceBench-Repair", "models/repair/repair_model"),
        ("GriceBench-DPO", "dpo_training_final_outcome")
    ]
    
    for repo_name, local_path in repos:
        repo_id = f"{args.username}/{repo_name}"
        print(f"Uploading to {repo_id}...")
        api.create_repo(repo_id=repo_id, exist_ok=True)
        if os.path.isdir(local_path):
            api.upload_folder(folder_path=local_path, repo_id=repo_id)
        else:
            api.upload_file(path_or_fileobj=local_path, path_in_repo=os.path.basename(local_path), repo_id=repo_id)
            
    print("Done!")

if __name__ == "__main__":
    main()
