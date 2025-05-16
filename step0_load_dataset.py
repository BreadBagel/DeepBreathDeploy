import os
import shutil
from pathlib import Path

# Path ng raw pneumonia dataset mo
pneumonia_path = r"C:\Users\User\Downloads\21176197\pneumonia\pneumonia"

# Gawa tayo ng bagong root folder para malinis
organized_root = r"C:\Users\User\Downloads\organized_cough_dataset"

# Target folder for pneumonia sounds
pneumonia_target = os.path.join(organized_root, "pneumonia")

# Step 1: Gawa ng organized folders
os.makedirs(pneumonia_target, exist_ok=True)

# Step 2: Copy lahat ng pneumonia files papunta sa organized folder
for filename in os.listdir(pneumonia_path):
    if filename.endswith(".mp3") or filename.endswith(".wav"):
        src_file = os.path.join(pneumonia_path, filename)
        dst_file = os.path.join(pneumonia_target, filename)
        shutil.copy2(src_file, dst_file)

print(f"✅ All pneumonia sounds copied to {pneumonia_target}")
