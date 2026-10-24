import os
import numpy as np
from PIL import Image

RAW_DIR = "/work/cvcs2026/blood_overflow/02_extracted/CholecSeg8k"
PROCESSED_DIR = "/work/cvcs2026/blood_overflow/03_processed/CholecSeg8k_processed"

os.makedirs(os.path.join(PROCESSED_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "masks"), exist_ok=True)

blood_pixel_value = 24  

video_folders = [f for f in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, f))]
processed_count = 0

for video in video_folders:
    video_path = os.path.join(RAW_DIR, video)
    segment_folders = [f for f in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, f))]
    
    for segment in segment_folders:
        seg_path = os.path.join(video_path, segment)
        if not os.path.exists(seg_path):
            continue
            
        files = os.listdir(seg_path)
        img_files = [f for f in files if f.endswith("_endo.png") and not f.endswith("mask.png")]
        
        for img_file in img_files:
            
            base_name = img_file.replace("_endo.png", "")
            
            watershed_mask_file = f"{base_name}_endo_watershed_mask.png"
            img_full_path = os.path.join(seg_path, img_file)
            watershed_full_path = os.path.join(seg_path, watershed_mask_file)
            
            if os.path.exists(watershed_full_path):
                mask_img = Image.open(watershed_full_path)
                mask_np = np.array(mask_img)
                
                if len(mask_np.shape) == 3:
                    mask_np = mask_np[:, :, 0]
                
                
                binary_mask = (mask_np == blood_pixel_value).astype(np.uint8) * 255
                               
                if np.sum(binary_mask) > 0:
                    
                    img_out_filename = f"{base_name}.png"
                    mask_out_filename = f"{base_name}_mask.png"
                    
                    img_out_path = os.path.join(PROCESSED_DIR, "images", img_out_filename)
                    mask_out_path = os.path.join(PROCESSED_DIR, "masks", mask_out_filename)
                    
                    Image.open(img_full_path).save(img_out_path)
                    Image.fromarray(binary_mask).save(mask_out_path)
                    processed_count += 1

print(f"Elaboration completed. Saved {processed_count} images/masks containing blood.")