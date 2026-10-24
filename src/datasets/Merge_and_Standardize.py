import os
import numpy as np
from PIL import Image

# Definiamo le sorgenti dei 3 dataset (modifica i percorsi se necessario in base a dove hai salvato i dati)
dataset_sources = [
    {
        "name": "CholecSeg8k",
        "img_dir": "/work/cvcs2026/blood_overflow/03_processed/CholecSeg8k_processed/images",
        "mask_dir": "/work/cvcs2026/blood_overflow/03_processed/CholecSeg8k_processed/masks"
    },
    {
        "name": "HemoSet",
        "img_dir": "/work/cvcs2026/blood_overflow/03_processed/HemoSet_processed/images",
        "mask_dir": "/work/cvcs2026/blood_overflow/03_processed/HemoSet_processed/masks"
    },
    {
        "name": "Rabbani",
        "img_dir": "/work/cvcs2026/blood_overflow/03_processed/Rabbani_processed/images",
        "mask_dir": "/work/cvcs2026/blood_overflow/03_processed/Rabbani_processed/masks"
    }
]


FINAL_DIR = "/work/cvcs2026/blood_overflow/03_processed/SurgPilot_Final_Dataset"
os.makedirs(os.path.join(FINAL_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(FINAL_DIR, "masks"), exist_ok=True)

global_counter = 0

print("STARTING MERGE AND STANDARDIZATION OF DATASETS...")

for ds in dataset_sources:
    img_path = ds["img_dir"]
    mask_path = ds["mask_dir"]
    
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f"Salto {ds['name']}: cartelle non trovate.")
        continue
        
    images = sorted([f for f in os.listdir(img_path) if f.endswith((".png", ".jpg"))])
    print(f"Processando {ds['name']}: trovate {len(images)} immagini.")
    
    for img_file in images:
        base_name, _ = os.path.splitext(img_file)
        
        possible_mask_names = [
            f"{base_name}_mask.png"
        ]
        
        src_mask_file = None
        for m_name in possible_mask_names:
            full_m_path = os.path.join(mask_path, m_name)
            if os.path.exists(full_m_path):
                src_mask_file = full_m_path
                break
                
        if src_mask_file is None:
            continue 
            
        src_img_file = os.path.join(img_path, img_file)
        
        
        new_filename = f"{global_counter:06d}.png"
        
        img = Image.open(src_img_file).convert("RGB")
        img.save(os.path.join(FINAL_DIR, "images", new_filename))
        
        
        mask_img = Image.open(src_mask_file)
        mask_np = np.array(mask_img)
        
        
        if len(mask_np.shape) == 3:
            non_black = np.sum(mask_np[:, :, :3], axis=-1) > 30
            binary_mask = non_black.astype(np.uint8) * 255
        else: 
            binary_mask = (mask_np > 0).astype(np.uint8) * 255
            
        
        Image.fromarray(binary_mask).save(os.path.join(FINAL_DIR, "masks", new_filename))
        
        global_counter += 1

print(f"Merging and standardization completed. Total unified image/mask pairs ready: {global_counter}")
print(f"Saved in: {FINAL_DIR}")