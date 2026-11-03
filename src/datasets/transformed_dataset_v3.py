import os
import shutil
import cv2

BLOOD_OVERFLOW_PATH = "/work/cvcs2026/blood_overflow"

SRC_IMAGES = os.path.join(BLOOD_OVERFLOW_PATH, '03_processed', 'Merged_Dataset', 'images')
SRC_MASKS  = os.path.join(BLOOD_OVERFLOW_PATH, '03_processed', 'Merged_Dataset', 'masks')

DEST_ROOT  = os.path.join(BLOOD_OVERFLOW_PATH, '03_processed', 'blooDet_inference_dataset_v3', 'test')

VIDEO_SEQUENCES = [
    ('cholec_seq01', 52, 1), ('cholec_seq02', 80, 1), ('cholec_seq03', 80, 1),
    ('cholec_seq04', 80, 1), ('cholec_seq05', 80, 1), ('cholec_seq06', 80, 1),
    ('cholec_seq07', 80, 1), ('cholec_seq08', 80, 1), ('cholec_seq09', 80, 1),
    ('hemoset_pig1', 153, 1), ('hemoset_pig2', 150, 1), ('hemoset_pig3', 50, 1),
    ('hemoset_pig4', 39, 1), ('hemoset_pig5', 131, 1), ('hemoset_pig6', 32, 1),
    ('hemoset_pig7', 64, 1), ('hemoset_pig9', 161, 1), ('hemoset_pig10', 162, 1),
    ('hemoset_pig11', 20, 1),
]

RABBANI_TOTAL = 751
CLIP_SIZE = 8

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

def apply_fast_transform(img, mask, dup_idx, h, w):
    if dup_idx == 0:
        return img, mask
    angle = (dup_idx - 3) * 1.5
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    t_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    t_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    beta = (dup_idx - 3) * 3
    t_img = cv2.convertScaleAbs(t_img, alpha=1.0, beta=beta)
    return t_img, t_mask

current_idx = 0
print(f"File totali trovati: {len(image_files)}")
print("Avvio elaborazione ottimizzata (shutil.copyfile per copie semplici)...\n")

for seq_name, num_frames_to_consume, duplicates in VIDEO_SEQUENCES:
    seq_img_path = os.path.join(dest_img_dir, seq_name)
    seq_mask_path = os.path.join(dest_mask_dir, seq_name)
    os.makedirs(seq_img_path, exist_ok=True)
    os.makedirs(seq_mask_path, exist_ok=True)

    files_to_process = image_files[current_idx : current_idx + num_frames_to_consume]

    for img_name in files_to_process:
        base_name = os.path.splitext(img_name)[0]
        src_img_path = os.path.join(SRC_IMAGES, img_name)
        src_mask_path = os.path.join(SRC_MASKS, img_name)

        if not os.path.exists(src_mask_path):
            src_mask_path = os.path.join(SRC_MASKS, f"{base_name}.png")

        # ------------------------------------------------------------------
        # 1. COPIA ULTRA-VELOCE DIRETTA SU DISCO PER CHOLEC ED HEMOSET (copie = 1)
        # ------------------------------------------------------------------
        if duplicates == 1:
            out_jpg_name = f"{base_name}.jpg"
            
            # Copia file immagine direttamente a livello di file system (senza aprirlo)
            if os.path.exists(src_img_path):
                shutil.copyfile(src_img_path, os.path.join(seq_img_path, out_jpg_name))
            
            # Copia file maschera
            if os.path.exists(src_mask_path):
                shutil.copyfile(src_mask_path, os.path.join(seq_mask_path, out_jpg_name))

        # ------------------------------------------------------------------
        # 2. TRASFORMAZIONE E GENERAZIONE SOLO PER RABBANI (copie > 1)
        # ------------------------------------------------------------------
        else:
            img = cv2.imread(src_img_path)
            mask = cv2.imread(src_mask_path)

            if img is not None and mask is not None:
                h, w = img.shape[:2]
                for dup_idx in range(duplicates):
                    out_jpg_name = f"{base_name}_{dup_idx:03d}.jpg"
                    t_img, t_mask = apply_fast_transform(img, mask, dup_idx, h, w)

                    cv2.imwrite(os.path.join(seq_img_path, out_jpg_name), t_img)
                    cv2.imwrite(os.path.join(seq_mask_path, out_jpg_name), t_mask)

    current_idx += num_frames_to_consume

print(f"\nCompletato con successo su:\n{DEST_ROOT}")