import os
from PIL import Image

RABBANI_DIR = "/work/cvcs2026/blood_overflow/02_extracted/Rabbani/bleeding_segmentation_v1p0"
PROCESSED_RABBANI_DIR = "/work/cvcs2026/blood_overflow/03_processed/Rabbani_processed"

os.makedirs(os.path.join(PROCESSED_RABBANI_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_RABBANI_DIR, "masks"), exist_ok=True)

imgs_dir = os.path.join(RABBANI_DIR, "images")
masks_dir = os.path.join(RABBANI_DIR, "masks")

processed_count = 0

if os.path.exists(imgs_dir) and os.path.exists(masks_dir):
    img_files = [f for f in os.listdir(imgs_dir) if f.endswith(".jpg")]
    
    for img_file in img_files:
        base_name = img_file.replace(".jpg", "") # es. "00001"
        mask_file = f"{base_name}.png"
        
        src_img_path = os.path.join(imgs_dir, img_file)
        src_mask_path = os.path.join(masks_dir, mask_file)
        
        if os.path.exists(src_mask_path):
            # Diamo un prefisso per evitare conflitti (es. rabbani_00001)
            new_name = f"rabbani_{base_name}"
            
            dst_img_path = os.path.join(PROCESSED_RABBANI_DIR, "images", f"{new_name}.png")
            dst_mask_path = os.path.join(PROCESSED_RABBANI_DIR, "masks", f"{new_name}_mask.png")
            
            # Convertiamo l'immagine JPG in PNG per uniformare tutto
            img = Image.open(src_img_path)
            img.save(dst_img_path, "PNG")
            
            # Copia e salva la maschera
            mask_img = Image.open(src_mask_path)
            mask_img.save(dst_mask_path)
            
            processed_count += 1

print(f"Elaboration completed. Saved {processed_count} images/masks from Rabbani dataset.")