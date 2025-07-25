import json
from tqdm import tqdm
from metric import TEDS
from .utils import preprocess, clean_html

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def unwrap_ground_truth(gt_data):
    if isinstance(gt_data, dict) and "image" in gt_data:
        return gt_data["image"]
    return gt_data

def build_dicts(pred_data, gt_data):
    pred_dict = {entry["filename"]: entry["html"] for entry in pred_data}
    gt_dict = {
        entry["filename"]: entry["text_html_table"]
        for entry in gt_data if entry.get("filename") and entry.get("text_html_table")
    }
    return pred_dict, gt_dict

def evaluate_teds(pred_dict, gt_dict):
    matched_files = sorted(set(pred_dict.keys()) & set(gt_dict.keys()))
    print(f"[INFO] Matched {len(matched_files)} files.")
    
    teds_structure = TEDS()
    teds_full = TEDS()

    scores_struc, scores_full = [], []

    for fname in tqdm(matched_files, desc="Calculating TEDS/TED"):
        try:
            pred_html = preprocess(pred_dict[fname])
            true_html = f"<html>{clean_html(gt_dict[fname])}</html>"
            scores_struc.append(teds_structure.evaluate(pred_html, true_html, is_structure=True))
            scores_full.append(teds_full.evaluate(pred_html, true_html, is_structure=False))
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    return scores_struc, scores_full
