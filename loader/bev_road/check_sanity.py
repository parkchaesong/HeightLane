import numpy as np
import json 
import glob
import os

heightmap_path = "/home/work/chase_data/openlane/heightmap_training"
heightmap_paths = glob.glob(os.path.join(heightmap_path, '**', '*.npy'), recursive=True)

print(f"총 {len(heightmap_paths)}개의 heightmap 파일을 찾았습니다.\n")
i=0
for hmap_path in heightmap_paths:
    try:
        hmap = np.load(hmap_path)
        hmap_mask = hmap[1,:,:]
        num_valid = hmap_mask.sum()
        if num_valid < 10:
            i+=1
            print(f"Loaded: {num_valid} | Shape: {hmap.shape}")
    except Exception as e:
        print(f"Error loading {hmap_path}: {e}")

print(i)