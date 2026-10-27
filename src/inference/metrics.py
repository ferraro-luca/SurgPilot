import os
import cv2
import numpy as np


# Cartella delle maschere predette da BlooDet (512x512)
PRED_MASK_DIR = '/work/cvcs2026/blood_overflow/05_results/blooDet_inference_results/videos-mask'

# Cartella delle maschere REALI (Ground Truth)
GT_MASK_DIR   = '/work/cvcs2026/blood_overflow/02_extracted/SurgBlood/SurgBlood-Dataset-test/test/videos-mask'


THRESHOLD = 127


def compute_metrics(pred_binary, gt_binary):
    """Calcola IoU, Dice, Precision e Recall tra due maschere binarie (bool)."""
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union        = np.logical_or(pred_binary, gt_binary).sum()
    pred_sum     = pred_binary.sum()
    gt_sum       = gt_binary.sum()

    # IoU (Intersection over Union / Jaccard Index)
    if union == 0:
        iou = 1.0 if (pred_sum == 0 and gt_sum == 0) else 0.0
    else:
        iou = intersection / union

    # Dice Coefficient (F1 Score)
    denominator = pred_sum + gt_sum
    if denominator == 0:
        dice = 1.0 if (pred_sum == 0 and gt_sum == 0) else 0.0
    else:
        dice = (2.0 * intersection) / denominator

    # Precision & Recall
    precision = (intersection / pred_sum) if pred_sum > 0 else (1.0 if gt_sum == 0 else 0.0)
    recall    = (intersection / gt_sum) if gt_sum > 0 else (1.0 if pred_sum == 0 else 0.0)

    return iou, dice, precision, recall



def main():
    if not os.path.exists(PRED_MASK_DIR):
        print(f"ERRORE: Cartella predizioni non trovata in: {PRED_MASK_DIR}")
        return

    if not os.path.exists(GT_MASK_DIR):
        print(f"ERRORE: Cartella Ground Truth non trovata in: {GT_MASK_DIR}")
        return

    video_folders = sorted(os.listdir(PRED_MASK_DIR))
    
    global_ious, global_dices = [], []
    global_precisions, global_recalls = [], []

    print("=" * 70)
    print(f"{'VIDEO':<25} | {'IoU (%)':<10} | {'Dice (%)':<10} | {'Precision':<10} | {'Recall':<10}")
    print("=" * 70)

    for video_name in video_folders:
        pred_video_path = os.path.join(PRED_MASK_DIR, video_name)
        gt_video_path   = os.path.join(GT_MASK_DIR, video_name)

        if not os.path.isdir(pred_video_path) or not os.path.exists(gt_video_path):
            continue

        video_ious, video_dices = [], []
        video_precisions, video_recalls = [], []

        frames = sorted([f for f in os.listdir(pred_video_path) if f.endswith(('.jpg', '.png'))])

        for frame_file in frames:
            gt_frame_path = os.path.join(gt_video_path, frame_file)
            
            if not os.path.exists(gt_frame_path):
                continue

            
            gt_img = cv2.imread(gt_frame_path, cv2.IMREAD_GRAYSCALE)
            if gt_img is None: continue
            h, w = gt_img.shape[:2]

            
            pred_img = cv2.imread(os.path.join(pred_video_path, frame_file), cv2.IMREAD_GRAYSCALE)
            if pred_img is None: continue
            pred_img_resized = cv2.resize(pred_img, (w, h), interpolation=cv2.INTER_NEAREST)

            
            gt_binary   = gt_img > THRESHOLD
            pred_binary = pred_img_resized > THRESHOLD

            
            iou, dice, prec, rec = compute_metrics(pred_binary, gt_binary)
            
            video_ious.append(iou)
            video_dices.append(dice)
            video_precisions.append(prec)
            video_recalls.append(rec)

        if video_ious:
            mean_iou  = np.mean(video_ious) * 100
            mean_dice = np.mean(video_dices) * 100
            mean_prec = np.mean(video_precisions) * 100
            mean_rec  = np.mean(video_recalls) * 100

            global_ious.extend(video_ious)
            global_dices.extend(video_dices)
            global_precisions.extend(video_precisions)
            global_recalls.extend(video_recalls)

            print(f"{video_name:<25} | {mean_iou:<10.2f} | {mean_dice:<10.2f} | {mean_prec:<10.2f} | {mean_rec:<10.2f}")

    print("=" * 70)
    if global_ious:
        print(f"{'RISULTATO GLOBALE':<25} | {np.mean(global_ious)*100:<10.2f} | {np.mean(global_dices)*100:<10.2f} | {np.mean(global_precisions)*100:<10.2f} | {np.mean(global_recalls)*100:<10.2f}")
    print("=" * 70)

if __name__ == '__main__':
    main()