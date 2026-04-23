from huggingface_hub import HfApi

def check_remote():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set.")
        return
    username = "Pushkar27"
    api = HfApi(token=token)
    
    repos = ["GriceBench-Detector", "GriceBench-Repair", "GriceBench-DPO"]
    
    for repo in repos:
        repo_id = f"{username}/{repo}"
        print(f"\nChecking {repo_id}:")
        files = api.list_repo_files(repo_id=repo_id)
        print(f"Files: {files}")
        
        # Check specific files
        if repo == "GriceBench-Detector":
            try:
                info = api.get_paths_info(repo_id=repo_id, paths=["pytorch_model.pt", "README.md"])
                for item in info:
                    print(f"  {item.path}: {item.size} bytes")
            except Exception as e:
                print(f"  Error checking paths: {e}")

if __name__ == "__main__":
    check_remote()
