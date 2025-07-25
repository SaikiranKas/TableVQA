import argparse
import csv
from docfm_evaluation.evaluate import load_json, unwrap_ground_truth, build_dicts, evaluate_teds

def main(args):
    preds = load_json(args.pred_path)
    raw_gt = load_json(args.gt_path)
    gts = unwrap_ground_truth(raw_gt)

    pred_dict, gt_dict = build_dicts(preds, gts)
    scores_struc, scores_full = evaluate_teds(pred_dict, gt_dict)

    avg_teds = sum(scores_struc) / len(scores_struc) if scores_struc else 0.0
    avg_ted = sum(scores_full) / len(scores_full) if scores_full else 0.0

    print(f"\n Average TEDS (structure only): {avg_teds:.4f}")
    print(f" Average TED  (full table):     {avg_ted:.4f}")

    if args.output_csv:
        with open(args.output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["avg_teds", "avg_ted"])
            writer.writeheader()
            writer.writerow({
                "avg_teds": round(avg_teds, 4),
                "avg_ted": round(avg_ted, 4)
            })
        print(f"\n[INFO] Saved results to: {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predicted JSON file")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to ground truth JSON file")
    parser.add_argument("--output_csv", type=str, default="teds_scores.csv", help="Output CSV path")
    args = parser.parse_args()
    main(args)
