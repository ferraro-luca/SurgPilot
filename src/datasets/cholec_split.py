import os
import shutil
import random

# 1. Percorso della cartella originale e della nuova cartella di output
SOURCE_DIR = "/work/cvcs2026/blood_overflow/03_processed/cholec_try/test"        # Cartella di partenza con videos-image e videos-mask
OUTPUT_DIR = "/work/cvcs2026/blood_overflow/03_processed/cholec_splitted"      # Nuova cartella che verrà creata

SRC_IMG_DIR = os.path.join(SOURCE_DIR, "videos-image")
SRC_MASK_DIR = os.path.join(SOURCE_DIR, "videos-mask")

# 2. Recupera l'elenco di tutte le sequenze
sequences = sorted([d for d in os.listdir(SRC_IMG_DIR) if os.path.isdir(os.path.join(SRC_IMG_DIR, d))])

# 3. Seed per fissare la riproducibilità dello split
random.seed(42)
random.shuffle(sequences)

# 4. Calcolo delle proporzioni (es. 60% Train, 20% Val, 20% Test)
total = len(sequences)
n_train = int(total * 0.60)
n_val = int(total * 0.20)

splits = {
    "train": sequences[:n_train],
    "val": sequences[n_train:n_train + n_val],
    "test": sequences[n_train + n_val:]
}

# 5. Creazione della nuova struttura di cartelle e copia dei file
print("Inizio la creazione del nuovo dataset diviso...")

for split_name, seq_list in splits.items():
    split_img_dir = os.path.join(OUTPUT_DIR, split_name, "videos-image")
    split_mask_dir = os.path.join(OUTPUT_DIR, split_name, "videos-mask")
    
    # Crea le cartelle di destinazione
    os.makedirs(split_img_dir, exist_ok=True)
    os.makedirs(split_mask_dir, exist_ok=True)
    
    for seq in seq_list:
        # Copia cartella immagini
        img_src = os.path.join(SRC_IMG_DIR, seq)
        img_dst = os.path.join(split_img_dir, seq)
        if os.path.exists(img_src) and not os.path.exists(img_dst):
            shutil.copytree(img_src, img_dst)
            
        # Copia cartella maschere
        mask_src = os.path.join(SRC_MASK_DIR, seq)
        mask_dst = os.path.join(split_mask_dir, seq)
        if os.path.exists(mask_src) and not os.path.exists(mask_dst):
            shutil.copytree(mask_src, mask_dst)

print("\n--- OPERAZIONE COMPLETATA ---")
print(f"Nuovo dataset creato in: {OUTPUT_DIR}")
print(f"Train: {len(splits['train'])} sequenze")
print(f"Val:   {len(splits['val'])} sequenze")
print(f"Test:  {len(splits['test'])} sequenze")