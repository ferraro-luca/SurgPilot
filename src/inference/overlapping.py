import cv2
import numpy as np

# 1. Carica il frame originale e la maschera binarizzata
frame_orig = cv2.imread('/work/cvcs2026/blood_overflow/03_processed/Merged_Dataset/images/000000.png')
#mask = cv2.imread('/work/cvcs2026/blood_overflow/04_results/blooDet_inference_results/videos-mask/cholec_seq01/000648.jpg', cv2.IMREAD_GRAYSCALE)
mask = cv2.imread('/work/cvcs2026/blood_overflow/03_processed/blooDet_inference_dataset/test/videos-mask/cholec_seq01/000000.jpg', cv2.IMREAD_GRAYSCALE)

# Ridimensiona la maschera alle dimensioni del frame originale se sono diverse
h, w = frame_orig.shape[:2]
mask_resized = cv2.resize(mask, (w, h))

# 2. Crea un layer rosso dove la maschera è bianca (> 127)
overlay = frame_orig.copy()
overlay[mask_resized > 2] = [255, 255, 255]  # BGR: Rosso

# 3. Fondi l'immagine originale con il layer rosso (trasparenza 30%)
result = cv2.addWeighted(overlay, 0.7, frame_orig, 0.6, 0)

# 4. Leggi le coordinate del punto dal file .txt e disegnalo
with open('/work/cvcs2026/blood_overflow/03_processed/blooDet_inference_dataset/test/videos-point/cholec_seq01/000000.txt', 'r') as f:
    norm_x, norm_y = map(float, f.readline().strip().split())

# De-normalizza sui pixel reali dell'immagine
px = int(norm_x * w)
py = int(norm_y * h)



cv2.circle(result, (px, py), radius=4, color=(255, 0, 0), thickness=-1)


# Salva l'overlay finale
cv2.imwrite('frame_con_sanguinamento_evidenziato.jpg', result)