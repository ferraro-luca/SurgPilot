import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# Percorsi di default basati sulla struttura di RabbaniPreprocessing.py
DEFAULT_INPUT_DIR = "/work/cvcs2026/blood_overflow/03_processed/Rabbani_processed"
DEFAULT_OUTPUT_DIR = "/work/cvcs2026/blood_overflow/03_processed/rabbani/test"

def split_rabbani(input_dir=DEFAULT_INPUT_DIR, output_dir=DEFAULT_OUTPUT_DIR):
    imgs_dir = os.path.join(input_dir, "images")
    masks_dir = os.path.join(input_dir, "masks")

    if not os.path.exists(imgs_dir) or not os.path.exists(masks_dir):
        raise FileNotFoundError(
            f"Cartelle 'images' o 'masks' non trovate in '{input_dir}'.\n"
            f"Assicurati che '{imgs_dir}' e '{masks_dir}' esistano."
        )

    out_images_dir = os.path.join(output_dir, "videos-image", "rabbani_clip")
    out_masks_dir = os.path.join(output_dir, "videos-mask", "rabbani_clip")

    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)

    # Estensioni supportate
    valid_img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = sorted([
        f for f in os.listdir(imgs_dir)
        if os.path.splitext(f)[1].lower() in valid_img_exts
    ])

    print(f"Trovate {len(img_files)} immagini in '{imgs_dir}'.")
    print(f"Salvataggio in:\n - Immagini: {out_images_dir}\n - Maschere: {out_masks_dir}")

    processed_count = 0
    missing_masks = 0

    for img_file in tqdm(img_files, desc="Elaborazione Rabbani"):
        base_name, _ = os.path.splitext(img_file)

        # Cerca la maschera corrispondente (es. .png come in RabbaniPreprocessing.py, o .jpg)
        src_mask_path = None
        for mask_ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            candidate = os.path.join(masks_dir, f"{base_name}{mask_ext}")
            if os.path.exists(candidate):
                src_mask_path = candidate
                break

        # Fallback nel caso la maschera abbia il suffisso '_mask'
        if src_mask_path is None:
            for mask_ext in [".png", ".jpg", ".jpeg"]:
                candidate = os.path.join(masks_dir, f"{base_name}_mask{mask_ext}")
                if os.path.exists(candidate):
                    src_mask_path = candidate
                    break

        if src_mask_path is None:
            missing_masks += 1
            continue

        src_img_path = os.path.join(imgs_dir, img_file)
        dst_img_path = os.path.join(out_images_dir, f"{base_name}.jpg")
        dst_mask_path = os.path.join(out_masks_dir, f"{base_name}.jpg")

        # Conversione e salvataggio immagine in JPG
        with Image.open(src_img_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(dst_img_path, "JPEG", quality=95)

        # Conversione e binarizzazione della maschera in JPG (nero=sfondo, bianco=sangue)
        with Image.open(src_mask_path) as mask_img:
            mask_rgb = mask_img.convert("RGB")
            mask_arr = np.array(mask_rgb)

            # Nelle maschere Rabbani il sangue è rosso e lo sfondo è nero.
            # Rileviamo i pixel del sangue (rosso dominante o attivo su sfondo nero)
            is_red = (mask_arr[:, :, 0] > 30) & (mask_arr[:, :, 0] > mask_arr[:, :, 1]) & (mask_arr[:, :, 0] > mask_arr[:, :, 2])
            is_red_variant = (mask_arr[:, :, 0] > 30) & (mask_arr[:, :, 1] < 50) & (mask_arr[:, :, 2] < 50)
            is_white = (mask_arr[:, :, 0] > 100) & (mask_arr[:, :, 1] > 100) & (mask_arr[:, :, 2] > 100)
            blood_pixels = is_red | is_red_variant | is_white

            # Creiamo la nuova maschera: sfondo nero (0, 0, 0) e sangue bianco (255, 255, 255)
            binary_mask = np.zeros_like(mask_arr, dtype=np.uint8)
            binary_mask[blood_pixels] = [255, 255, 255]

            out_mask_img = Image.fromarray(binary_mask, mode="RGB")
            # Salva con qualità massima per preservare i dettagli della maschera
            out_mask_img.save(dst_mask_path, "JPEG", quality=100, subsampling=0)

        processed_count += 1

    print("\nElaborazione completata!")
    print(f"- Coppie immagine/maschera elaborate con successo: {processed_count}")
    if missing_masks > 0:
        print(f"- Immagini saltate per maschera non trovata: {missing_masks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dividi e converti il dataset Rabbani in videos-image e videos-mask in formato JPG."
    )
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f"Cartella principale contenente 'images' e 'masks' (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Cartella di destinazione (default: {DEFAULT_OUTPUT_DIR})"
    )

    args = parser.parse_args()
    split_rabbani(input_dir=args.input_dir, output_dir=args.output_dir)
