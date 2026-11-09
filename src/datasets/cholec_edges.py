import os
import cv2
import numpy as np

# Path alla cartella principale del nuovo dataset diviso
DATASET_DIR = "/work/cvcs2026/blood_overflow/03_processed/cholec_splitted"  # Sostituisci col tuo percorso

# Consideriamo SOLO lo split di train
train_mask_root = os.path.join(DATASET_DIR, "train", "videos-mask")
train_edge_root = os.path.join(DATASET_DIR, "train", "videos-edge")

if os.path.exists(train_mask_root):
    print("Generazione 'videos-edge' in corso SOLO per lo split di TRAIN...")
    
    # Kernel 3x3 per l'estrazione del contorno
    kernel = np.ones((3, 3), np.uint8)
    
    # Scorre ogni sequenza video nel train
    for seq_name in os.listdir(train_mask_root):
        seq_mask_dir = os.path.join(train_mask_root, seq_name)
        if not os.path.isdir(seq_mask_dir):
            continue
            
        seq_edge_dir = os.path.join(train_edge_root, seq_name)
        os.makedirs(seq_edge_dir, exist_ok=True)
        
        # Scorre tutte le maschere della sequenza
        for mask_file in os.listdir(seq_mask_dir):
            if mask_file.endswith(('.png', '.jpg', '.jpeg')):
                mask_path = os.path.join(seq_mask_dir, mask_file)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if mask is None:
                    continue
                
                # Gradiente Morfologico: estrazione contorno binario
                edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
                
                # Salva l'immagine dell'edge
                edge_path = os.path.join(seq_edge_dir, mask_file)
                cv2.imwrite(edge_path, edge)

    print("\nGenerazione delle sotto-cartelle 'videos-edge' completata con successo nel TRAIN!")
else:
    print(f"Errore: Cartella non trovata in {train_mask_root}")