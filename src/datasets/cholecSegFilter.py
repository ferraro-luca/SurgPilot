import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Percorsi sul cluster CINECA
RAW_DIR = "/work/cvcs2026/blood_overflow/02_extracted/CholecSeg8k"
PROCESSED_DIR = "/work/cvcs2026/blood_overflow/03_processed/cholec_processed"

# Creazione delle cartelle di output
os.makedirs(os.path.join(PROCESSED_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "masks"), exist_ok=True)

blood_class_id = 7  # Classe 7: Blood dalla tabella CholecSeg8k[cite: 8]

# Scansiona tutte le cartelle dei video (es. video01, video09, ecc.)
video_folders = [f for f in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, f))]

for video in video_folders:
    video_path = os.path.join(RAW_DIR, video)
    segment_folders = [f for f in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, f))]
    
    for segment in segment_folders:
        seg_path = os.path.join(video_path, segment)
        
        # Trova tutti i file immagine originali (terminano con _endo.png)
        files = os.listdir(seg_path)
        img_files = [f for f in files if f.endswith("_endo.png") and not f.endswith("mask.png")]
        
        for img_file in img_files:
            # Ricostruisci il nome della maschera corrispondente (_endo_mask.png)
            base_name = img_file.replace("_endo.png", "")
            mask_file = f"{base_name}_endo_mask.png"
            
            img_full_path = os.path.join(seg_path, img_file)
            mask_full_path = os.path.join(seg_path, mask_file)
            
            if os.path.exists(mask_full_path):
                # Carica la maschera
                mask_img = Image.open(mask_full_path)
                mask_np = np.array(mask_img)
                
                # Binarizzazione: 1 dove c'è la classe 7 (sangue), 0 altrove[cite: 8]
                binary_mask = (mask_np == blood_class_id).astype(np.uint8) * 255
                
                # Salviamo solo se nel frame è effettivamente presente del sangue (opzionale ma utile per filtrare i frame vuoti)
                if np.sum(binary_mask) > 0:
                    unique_name = f"{video}_{segment}_{base_name}"
                    
                    # Copia/Salva immagine originale e nuova maschera binaria
                    img_out_path = os.path.join(PROCESSED_DIR, "images", f"{unique_name}.png")
                    mask_out_path = os.path.join(PROCESSED_DIR, "masks", f"{unique_name}.png")
                    
                    Image.open(img_full_path).save(img_out_path)
                    Image.fromarray(binary_mask).save(mask_out_path)

print("Elaborazione e filtraggio di CholecSeg8k completati con successo!")