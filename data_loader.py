import kagglehub
import shutil
import os

# Download to cache first
path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")

# Copy entire directory tree to your desired location
destination = "."
shutil.copytree(path, destination, dirs_exist_ok=True)

print("Dataset copied to:", destination)
