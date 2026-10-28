import os
from PIL import Image

BLOOD_OVERFLOW_PATH = "/work/cvcs2026/blood_overflow"

SRC_IMAGES = os.path.join(BLOOD_OVERFLOW_PATH, '03_processed', 'Merged_Dataset', 'images')
SRC_MASKS  = os.path.join(BLOOD_OVERFLOW_PATH, '03_processed', 'Merged_Dataset', 'masks')

DEST_ROOT  = os.path.join(BLOOD_OVERFLOW_PATH, '03_processed', 'blooDet_inference_dataset_v2', 'test')

# Formato: (nome_sequenza, frame_sorgente_da_consumare, copie_per_frame)
VIDEO_SEQUENCES = [
    ('cholec_seq01', 52, 1),
    ('cholec_seq02', 80, 1),
    ('cholec_seq03', 80, 1),
    ('cholec_seq04', 80, 1),
    ('cholec_seq05', 80, 1),
    ('cholec_seq06', 80, 1),
    ('cholec_seq07', 80, 1),
    ('cholec_seq08', 80, 1),
    ('cholec_seq09', 80, 1),
    ('hemoset_pig1', 153, 1),
    ('hemoset_pig2', 150, 1),
    ('hemoset_pig3', 50, 1),
    ('hemoset_pig4', 39, 1),
    ('hemoset_pig5', 131, 1),
    ('hemoset_pig6', 32, 1),
    ('hemoset_pig7', 64, 1),
    ('hemoset_pig9', 161, 1),
    ('hemoset_pig10', 162, 1),
    ('hemoset_pig11', 20, 1),
]

RABBANI_TOTAL = 751
CLIP_SIZE = 8

# Generazione logica Rabbani: 751 cartelle distinte, ognuna consuma 1 frame e lo duplica 8 volte
for i in range(1, RABBANI_TOTAL + 1):
    VIDEO_SEQUENCES.append((f'rabbani_clip{i:03d}', 1, CLIP_SIZE))

dest_img_dir = os.path.join(DEST_ROOT, 'videos-image')
dest_mask_dir = os.path.join(DEST_ROOT, 'videos-mask')
os.makedirs(dest_img_dir, exist_ok=True)
os.makedirs(dest_mask_dir, exist_ok=True)

image_files = sorted(
    [f for f in os.listdir(SRC_IMAGES) if f.endswith(('.png', '.jpg', '.jpeg'))],
    key=lambda x: int(os.path.splitext(x)[0])
)

current_idx = 0
print(f"File totali trovati in origine: {len(image_files)}")
print(f"Origine: {SRC_IMAGES}")
print(f"Destinazione (Formato JPG): {DEST_ROOT}\n")

for seq_name, num_frames_to_consume, duplicates in VIDEO_SEQUENCES:
    seq_img_path = os.path.join(dest_img_dir, seq_name)
    seq_mask_path = os.path.join(dest_mask_dir, seq_name)
    os.makedirs(seq_img_path, exist_ok=True)
    os.makedirs(seq_mask_path, exist_ok=True)
    
    files_to_process = image_files[current_idx : current_idx + num_frames_to_consume]
    
    print(f"-> Elaborazione '{seq_name}' ({len(files_to_process)} frame sorgente, {duplicates} copie ciascuno)...")
    
    for img_name in files_to_process:
        base_name = os.path.splitext(img_name)[0]
        
        src_img_path = os.path.join(SRC_IMAGES, img_name)
        src_mask_path = os.path.join(SRC_MASKS, img_name)
        
        if not os.path.exists(src_mask_path):
            src_mask_path = os.path.join(SRC_MASKS, f"{base_name}.png")

        # Carica le immagini in memoria e chiude subito i file originali
        img_obj, mask_obj = None, None
        
        if os.path.exists(src_img_path):
            with Image.open(src_img_path) as img:
                img_obj = img.convert('RGB') if img.mode != 'RGB' else img.copy()
                
        if os.path.exists(src_mask_path):
            with Image.open(src_mask_path) as mask:
                mask_obj = mask.convert('RGB') if mask.mode != 'RGB' else mask.copy()
        else:
            print(f"   [ATTENZIONE] Maschera non trovata per: {img_name}")

        # Salva le copie richieste
        for dup_idx in range(duplicates):
            if duplicates == 1:
                out_jpg_name = f"{base_name}.jpg"
            else:
                out_jpg_name = f"{base_name}_{dup_idx:03d}.jpg"

            if img_obj:
                img_obj.save(os.path.join(seq_img_path, out_jpg_name), 'JPEG', quality=95)
            if mask_obj:
                mask_obj.save(os.path.join(seq_mask_path, out_jpg_name), 'JPEG', quality=95)
                
    current_idx += num_frames_to_consume

print(f"\nCompleted. all images and masks have been converted to JPG and saved in:\n{DEST_ROOT}")