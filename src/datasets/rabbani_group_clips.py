import os
import re
import shutil
import argparse
from tqdm import tqdm

DEFAULT_DATASET_DIR = "/work/cvcs2026/blood_overflow/03_processed/rabbani/test"
OUTPUT_DATASET_DIR = "/work/cvcs2026/blood_overflow/03_processed/rabbani_clips/test"

def natural_sort_key(s):
    """Ordina in modo naturale (es. img1, img2, ..., img10 anziché img1, img10, img2)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def find_clip_source_dir(base_dir, sub_name):
    """Trova la cartella sorgente, gestendo sia 'sub_name/rabbani_clip' che 'sub_name' diretto."""
    clip_dir = os.path.join(base_dir, sub_name, "rabbani_clip")
    if os.path.isdir(clip_dir):
        return clip_dir
    direct_dir = os.path.join(base_dir, sub_name)
    if os.path.isdir(direct_dir):
        return direct_dir
    return None

def group_rabbani_clips(
    dataset_dir=DEFAULT_DATASET_DIR,
    output_dir=None,
    num_copies=8,
    start_frame_idx=0
):
    if output_dir is None:
        output_dir = dataset_dir

    # Individua cartelle sorgente per immagini e maschere
    imgs_src_dir = find_clip_source_dir(dataset_dir, "videos-image")
    masks_src_dir = find_clip_source_dir(dataset_dir, "videos-mask")

    if not imgs_src_dir or not os.path.isdir(imgs_src_dir):
        raise FileNotFoundError(
            f"Impossibile trovare la cartella delle immagini in '{dataset_dir}'. "
            f"Verifica che esista '{os.path.join(dataset_dir, 'videos-image')}' o "
            f"'{os.path.join(dataset_dir, 'videos-image', 'rabbani_clip')}'."
        )

    has_masks = masks_src_dir and os.path.isdir(masks_src_dir)
    if not has_masks:
        print(f"[ATTENZIONE] Cartella maschere non trovata in '{dataset_dir}'. Verranno elaborate solo le immagini.")

    # Trova tutti i file immagine
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = [
        f for f in os.listdir(imgs_src_dir)
        if os.path.splitext(f)[1].lower() in valid_exts and os.path.isfile(os.path.join(imgs_src_dir, f))
    ]
    img_files.sort(key=natural_sort_key)

    total_images = len(img_files)
    print(f"Trovate {total_images} immagini in: {imgs_src_dir}")
    print(f"Creazione di {total_images} cartelle 'rabbani_clip<N>' con {num_copies} copie ciascuna.")

    # Cartelle radice di destinazione
    dst_images_root = os.path.join(output_dir, "videos-image")
    dst_masks_root = os.path.join(output_dir, "videos-mask")
    os.makedirs(dst_images_root, exist_ok=True)
    if has_masks:
        os.makedirs(dst_masks_root, exist_ok=True)

    # Elenco delle maschere disponibili per lookup rapido
    mask_files_map = {}
    if has_masks:
        for mf in os.listdir(masks_src_dir):
            if os.path.isfile(os.path.join(masks_src_dir, mf)):
                base, _ = os.path.splitext(mf)
                mask_files_map[base] = mf

    for clip_idx, img_file in enumerate(tqdm(img_files, desc="Creazione clip"), start=1):
        clip_name = f"rabbani_clip{clip_idx}"
        img_base_name, img_ext = os.path.splitext(img_file)

        clip_img_dir = os.path.join(dst_images_root, clip_name)
        os.makedirs(clip_img_dir, exist_ok=True)

        src_img_path = os.path.join(imgs_src_dir, img_file)

        # Cerca la maschera corrispondente
        src_mask_path = None
        clip_mask_dir = None
        if has_masks:
            mask_filename = mask_files_map.get(img_base_name)
            if mask_filename:
                src_mask_path = os.path.join(masks_src_dir, mask_filename)
                clip_mask_dir = os.path.join(dst_masks_root, clip_name)
                os.makedirs(clip_mask_dir, exist_ok=True)

        # Genera le copie del frame (es. 00000.jpg, 00001.jpg, ... o frame_0.jpg)
        for copy_idx in range(num_copies):
            frame_num = start_frame_idx + copy_idx
            frame_filename = f"{frame_num:05d}{img_ext}"

            # Copia immagine
            dst_img_path = os.path.join(clip_img_dir, frame_filename)
            shutil.copy2(src_img_path, dst_img_path)

            # Copia maschera (se presente)
            if src_mask_path and clip_mask_dir:
                mask_ext = os.path.splitext(src_mask_path)[1]
                mask_frame_filename = f"{frame_num:05d}{mask_ext}"
                dst_mask_path = os.path.join(clip_mask_dir, mask_frame_filename)
                shutil.copy2(src_mask_path, dst_mask_path)

    print("\nRaggruppamento completato con successo!")
    print(f"- Cartelle create: {total_images} (da rabbani_clip1 a rabbani_clip{total_images})")
    print(f"- Frame per cartella: {num_copies}")
    print(f"- Destinazione immagini: {dst_images_root}")
    if has_masks:
        print(f"- Destinazione maschere: {dst_masks_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Raggruppa le immagini/maschere di Rabbani in cartelle clip contenenti copie della stessa immagine."
    )
    parser.add_argument(
        "--dataset_dir", "-d",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help=f"Cartella principale contenente 'videos-image' e 'videos-mask' (default: {DEFAULT_DATASET_DIR})"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=OUTPUT_DATASET_DIR,
        help="Cartella di destinazione (default: stessa di --dataset_dir)"
    )
    parser.add_argument(
        "--num_copies", "-n",
        type=int,
        default=8,
        help="Numero di copie per cartella clip (default: 8)"
    )
    parser.add_argument(
        "--start_frame_idx",
        type=int,
        default=0,
        help="Indice di partenza per i frame (default: 0 -> 00000.jpg, 00001.jpg, ...)"
    )

    args = parser.parse_args()
    group_rabbani_clips(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        num_copies=args.num_copies,
        start_frame_idx=args.start_frame_idx
    )
