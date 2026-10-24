import os
import shutil
from PIL import Image

HEMO_DIR = "/work/cvcs2026/blood_overflow/02_extracted/HemoSet/HemoSet_Labels"
PROCESSED_HEMO_DIR = "/work/cvcs2026/blood_overflow/03_processed/HemoSet_processed"

os.makedirs(os.path.join(PROCESSED_HEMO_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_HEMO_DIR, "masks"), exist_ok=True)

processed_count = 0


for i in range(1, 12):
    pig_name = f"pig{i}"
    pig_path = os.path.join(HEMO_DIR, pig_name)
    
    if not os.path.exists(pig_path):
        continue
        
    imgs_dir = os.path.join(pig_path, "imgs")
    labels_dir = os.path.join(pig_path, "labels")
    
    if not os.path.exists(imgs_dir) or not os.path.exists(labels_dir):
        continue
        
    img_files = [f for f in os.listdir(imgs_dir) if f.endswith(".png")]
    
    for img_file in img_files:
        base_name = img_file.replace(".png", "") 
        mask_file = f"{base_name}_mask.png"
        
        src_img_path = os.path.join(imgs_dir, img_file)
        src_mask_path = os.path.join(labels_dir, mask_file)
        
        if os.path.exists(src_mask_path):
            
            unique_filename = f"{pig_name}_{base_name}.png"
            unique_maskname = f"{pig_name}_{base_name}_mask.png"
            
            dst_img_path = os.path.join(PROCESSED_HEMO_DIR, "images", unique_filename)
            dst_mask_path = os.path.join(PROCESSED_HEMO_DIR, "masks", unique_maskname)
            
            shutil.copy(src_img_path, dst_img_path)
            shutil.copy(src_mask_path, dst_mask_path)
            
            processed_count += 1

print(f"Elaboration completed. Saved {processed_count} images/masks from HemoSet.")