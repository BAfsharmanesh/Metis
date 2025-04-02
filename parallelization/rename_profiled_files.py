import os
import re
import shutil
from parallelization.utils import manipulate_write_new_file, create_dummy_profile

def copy_2_new_path(old_path, new_path):
    if (old_path == new_path) or os.path.exists(new_path):
        return
    # copy the file, keep the old file
    shutil.copyfile(old_path, new_path)
    print(f"Copied: {old_path} -> {new_path}")


def rename_files(base_dir):
    # Walk through the directory structure
    for root, _, files in os.walk(base_dir):
        for file in files:
            # Match files with the pattern "<prefix>_<size>_<rest>.json"
            match = re.match(r"(.*)_(\d+(?:\.\d+)?[MB])_(DeviceType\..*\.json)", file)
            if match:
                prefix, size, rest = match.groups()
                
                # Extract the base folder name (e.g., "wideresnet" -> "1B")
                base_folder = size
                
                # Construct the new file path
                old_path = os.path.join(root, file)
                new_folder = os.path.join(root, size)  # e.g., "profile/merged_wresnet/1B"
                os.makedirs(new_folder, exist_ok=True)  # Create the folder if it doesn't exist
                new_path = os.path.join(new_folder, rest)  # e.g., "DeviceType.A6000_tp1_bs4.json"

                manipulate_write_new_file(old_path, new_path) # apply corrections (if needed) to the file and write it to the new path
                create_dummy_profile(old_path, new_path) # create a dummy profile for the new file

            else:
                print(f"Skipped: {file}")

if __name__ == "__main__":
    base_directory = "test"  # Change this to your base directory
    rename_files(base_directory)
