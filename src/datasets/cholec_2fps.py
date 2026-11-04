import os
import shutil

BASE_PATH = "/work/cvcs2026/blood_overflow/03_processed"

SRC_DIR  = os.path.join(BASE_PATH, "cholec_try", "test")
DEST_DIR = os.path.join(BASE_PATH, "cholec_try_2fps", "test")


STEP = 12

def downsample_dataset():
    if not os.path.exists(SRC_DIR):
        print(f"ERRORE: Cartella sorgente non trovata in: {SRC_DIR}")
        return

    # Sottocartelle principali (videos-image e videos-mask)
    subfolders = ['videos-image', 'videos-mask']

    for subfolder in subfolders:
        src_sub_path = os.path.join(SRC_DIR, subfolder)
        dest_sub_path = os.path.join(DEST_DIR, subfolder)

        if not os.path.exists(src_sub_path):
            print(f"[ATTENZIONE] Sottocartella mancante: {src_sub_path}")
            continue

        os.makedirs(dest_sub_path, exist_ok=True)

        # Scorre tutte le sotto-cartelle dei video (clip)
        seq_folders = sorted(os.listdir(src_sub_path))

        for seq_name in seq_folders:
            src_seq_dir  = os.path.join(src_sub_path, seq_name)
            dest_seq_dir = os.path.join(dest_sub_path, seq_name)

            if not os.path.isdir(src_seq_dir):
                continue

            os.makedirs(dest_seq_dir, exist_ok=True)

            # Prende tutti i frame ordinati
            all_frames = sorted([
                f for f in os.listdir(src_seq_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

            # Campionamento a passo 12 (1 frame ogni 12)
            selected_frames = all_frames[::STEP]

            print(f"-> Sequenza '{seq_name}' [{subfolder}]: presi {len(selected_frames)} frame su {len(all_frames)} totali")

            # Copia veloce a basso livello dei file campionati
            for frame_name in selected_frames:
                src_file = os.path.join(src_seq_dir, frame_name)
                dest_file = os.path.join(dest_seq_dir, frame_name)
                shutil.copyfile(src_file, dest_file)

    print("\n" + "=" * 70)
    print(f"Completato! Nuovo dataset ridotto a 2fps salvato in:\n{DEST_DIR}")
    print("=" * 70)

if __name__ == '__main__':
    downsample_dataset()