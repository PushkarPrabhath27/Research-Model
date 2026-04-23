import os
import sys
from huggingface_hub import HfApi

def upload_project():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set.")
        return
    username = "Pushkar27"
    api = HfApi(token=token)
    
    try:
        user_info = api.whoami()
        print(f"Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"Authentication failed: {e}")
        return

    # 1. GriceBench-Detector
    detector_repo = f"{username}/GriceBench-Detector"
    print(f"\nProcessing {detector_repo}...")
    api.create_repo(repo_id=detector_repo, exist_ok=True)
    
    # Upload README
    api.upload_file(
        path_or_fileobj="MODEL_CARD_DETECTOR.md",
        path_in_repo="README.md",
        repo_id=detector_repo
    )
    # Upload Weights (Renaming for standard HF usage)
    api.upload_file(
        path_or_fileobj="best_model_v2.pt",
        path_in_repo="pytorch_model.pt",
        repo_id=detector_repo
    )
    # Upload Temperatures
    api.upload_file(
        path_or_fileobj="temperatures.json",
        path_in_repo="temperatures.json",
        repo_id=detector_repo
    )
    print(f"DONE: {detector_repo} completed.")

    # 2. GriceBench-Repair
    repair_repo = f"{username}/GriceBench-Repair"
    print(f"\nProcessing {repair_repo}...")
    api.create_repo(repo_id=repair_repo, exist_ok=True)
    
    # Upload README
    api.upload_file(
        path_or_fileobj="MODEL_CARD_REPAIR.md",
        path_in_repo="README.md",
        repo_id=repair_repo
    )
    # Upload Model Folder
    api.upload_folder(
        folder_path="models/repair/repair_model",
        repo_id=repair_repo,
        ignore_patterns=["README.md"] # Avoid overwriting the one we just uploaded
    )
    print(f"DONE: {repair_repo} completed.")

    # 3. GriceBench-DPO
    dpo_repo = f"{username}/GriceBench-DPO"
    print(f"\nProcessing {dpo_repo}...")
    api.create_repo(repo_id=dpo_repo, exist_ok=True)
    
    # Upload README
    api.upload_file(
        path_or_fileobj="MODEL_CARD_DPO.md",
        path_in_repo="README.md",
        repo_id=dpo_repo
    )
    # Upload Adapter Folder
    api.upload_folder(
        folder_path="dpo_training_final_outcome",
        repo_id=dpo_repo,
        ignore_patterns=["README.md"]
    )
    print(f"DONE: {dpo_repo} completed.")

    print("\nALL UPLOADS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    upload_project()
