from azureml.fsspec import AzureMachineLearningFileSystem
import tarfile
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
archive_path = os.path.join(current_dir, "data", "archives", "v1.0-mini.tgz")
if not os.path.exists(archive_path):
    # 1. Your specific URI
    uri = "azureml://subscriptions/69b95eae-0191-45b3-8bac-bda99d889c0e/resourcegroups/InferenceComputing/workspaces/OpenEMMAinf/datastores/workspaceblobstore/paths/"

    # 2. Define the remote file path and local destination
    remote_file_path = "UI/2026-02-08_122038_UTC/v1.0-mini.tgz"
    local_destination = "data/archives/" # This will create a folder in your current directory

    # 3. Initialize the filesystem
    fs = AzureMachineLearningFileSystem(uri)

    # 4. Execute the download
    print(f"Downloading {remote_file_path}...")
    fs.download(rpath=remote_file_path, lpath=local_destination, recursive=False)

    print("Download complete!")
    print(f"File location: {os.path.abspath(local_destination)}")

# 1. Define paths relative to this script

extract_path = os.path.join(current_dir, "data", "nuscenes_mini")

# 2. Ensure the source file exists
if not os.path.exists(archive_path):
    print(f"Error: Could not find the file at {archive_path}")
else:
    print(f"Extracting {archive_path} to {extract_path}...")
    
    # 3. Open and extract the .tgz file
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    
    print("Extraction successful!")