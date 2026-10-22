import os
import numpy as np
from PIL import Image
from tqdm import tqdm

RAW_DIR = "/work/cvcs2026/blood_overflow/02_extracted/CholecSeg8k"
PROCESSED_DIR = "/work/cvcs2026/blood_overflow/03_processed/CholecSeg8k_processed"

os.makedirs(os.path.join(PROCESSED_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "masks"), exist_ok=True)

blood_class_id = 7  

video_folders = [f for f in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, f))]

for video in video_folders:
    video_path = os.path.join(RAW_DIR, video)
    segment_folders = [f for f in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, f))]
    
    for segment in segment_folders:
        seg_path = os.path.join(video_path, segment)
            
        files = os.listdir(seg_path)
        img_files = [f for f in files if f.endswith("_endo.png") and not f.endswith("mask.png")]
        
        for img_file in img_files:
            
            base_name = img_file.replace("_endo.png", "")
            mask_file = f"{base_name}_endo_mask.png"
            
            img_full_path = os.path.join(seg_path, img_file)
            mask_full_path = os.path.join(seg_path, mask_file)
            
            if os.path.exists(mask_full_path):
                
                mask_img = Image.open(mask_full_path)
                mask_np = np.array(mask_img)
                               
                binary_mask = (mask_np == blood_class_id).astype(np.uint8) * 255
                               
                if np.sum(binary_mask) > 0:
                    unique_name = f"{video}_{segment}_{base_name}"
                    
                    img_out_path = os.path.join(PROCESSED_DIR, "images", f"{unique_name}.png")
                    mask_out_path = os.path.join(PROCESSED_DIR, "masks", f"{unique_name}.png")
                    
                    Image.open(img_full_path).save(img_out_path)
                    Image.fromarray(binary_mask).save(mask_out_path)

print("Elaboration completed. Processed images and masks are saved in the '03_processed/CholecSeg8k_processed' directory.")