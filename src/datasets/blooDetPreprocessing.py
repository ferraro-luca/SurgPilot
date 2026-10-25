import os
from PIL import Image

BLOOD_OVERFLOW_PATH = "/work/cvcs2026/blood_overflow"


SRC_IMAGES = os.path.join(BLOOD_OVERFLOW_PATH, '03_processed', 'Merged_Dataset', 'images')
SRC_MASKS  = os.path.join(BLOOD_OVERFLOW_PATH, '03_processed', 'Merged_Dataset', 'masks')


DEST_ROOT  = os.path.join(BLOOD_OVERFLOW_PATH, '03_processed', 'blooDet_inference_dataset', 'test')


VIDEO_SEQUENCES = [
    
    ('cholec_seq01', 692),
    ('hemoset_pig1', 153),
    ('hemoset_pig2', 150),
    ('hemoset_pig3', 50),
    ('hemoset_pig4', 39),
    ('hemoset_pig5', 131),
    ('hemoset_pig6', 32),
    ('hemoset_pig7', 64),
    ('hemoset_pig9', 161),
    ('hemoset_pig10', 162),
    ('hemoset_pig11', 20),
]


RABBANI_TOTAL = 751
CLIP_SIZE = 30
rem_frames = RABBANI_TOTAL
clip_num = 1

while rem_frames > 0:
    current_len = min(CLIP_SIZE, rem_frames)
    VIDEO_SEQUENCES.append((f'rabbani_clip{clip_num:02d}', current_len))
    rem_frames -= current_len
    clip_num += 1


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

for seq_name, num_frames in VIDEO_SEQUENCES:
    seq_img_path = os.path.join(dest_img_dir, seq_name)
    seq_mask_path = os.path.join(dest_mask_dir, seq_name)
    os.makedirs(seq_img_path, exist_ok=True)
    os.makedirs(seq_mask_path, exist_ok=True)
    
    files_to_process = image_files[current_idx : current_idx + num_frames]
    
    print(f"-> Elaborazione sequenza '{seq_name}' ({len(files_to_process)} frame)...")
    
    for img_name in files_to_process:
        base_name = os.path.splitext(img_name)[0]
        out_jpg_name = f"{base_name}.jpg"
        
        src_img_path = os.path.join(SRC_IMAGES, img_name)
        
        src_mask_path = os.path.join(SRC_MASKS, img_name)
        if not os.path.exists(src_mask_path):
            src_mask_path = os.path.join(SRC_MASKS, f"{base_name}.png")

       
        if os.path.exists(src_img_path):
            with Image.open(src_img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(os.path.join(seq_img_path, out_jpg_name), 'JPEG', quality=95)

        
        if os.path.exists(src_mask_path):
            with Image.open(src_mask_path) as mask:
                if mask.mode != 'RGB':
                    mask = mask.convert('RGB')
                mask.save(os.path.join(seq_mask_path, out_jpg_name), 'JPEG', quality=95)
        else:
            print(f"   [ATTENZIONE] Maschera non trovata per: {img_name}")
            
    current_idx += num_frames

print(f"\nCompleted. all images and masks have been converted to JPG and saved in:\n{DEST_ROOT}")