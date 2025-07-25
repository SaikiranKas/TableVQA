import torch
import json
import os
from tqdm import tqdm
import Levenshtein
from transformers import AutoTokenizer, LlamaForCausalLM

# === Device Setup ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Model Wrapper ===
class TableVQAModel(torch.nn.Module):
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        super().__init__()
        print(f"Loading model: {model_name}")
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"  # Use automatic GPU placement
        )
        self.model.eval()

    def forward(self, input_ids, labels):
        outputs = self.model(input_ids=input_ids, labels=labels)
        return outputs.loss, outputs.logits

def extract_answer(decoded_output, input_text):
    decoded_output = decoded_output.lower()
    if "### answer:" in decoded_output:
        start = decoded_output.find("### answer:") + len("### answer:")
        end = decoded_output.find("###", start)
        return decoded_output[start:end].strip() if end != -1 else decoded_output[start:].strip()
    else:
        return decoded_output.replace(input_text.lower(), "").strip()

# === Main Evaluation Logic ===
def main():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    checkpoint_path = "/llama8bresults/tablevqa_epoch4.pth"  # update epoch number as needed
    test_path = "src/model/combined_wtq_html_otsl_test.json"

    # === Tokenizer and Model ===
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = TableVQAModel(model_name=model_name)

    # Load trained weights
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    base_model.load_state_dict(new_state_dict)
    base_model = base_model.to(device)
    model = base_model.model  # unwrap inner model for `.generate`

    print(f"Loaded model from: {checkpoint_path}")

    # === Load Test Data ===
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples.")

    exact_match = 0
    similar_match = 0
    total = 0
    predictions = []

    for idx, entry in enumerate(tqdm(test_data)):
        question = entry["question"]
        ground_truth = entry["answer_text"].strip().lower()
        table_html = entry["otsl"]

        input_text = f"""### Instruction:
        Given the following table, answer the question in one word or short phrase. Do not provide an explanation.

        ### Table:
        {table_html}

        ### Question:
        {question}

        ### Answer:"""

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=4096
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                do_sample=False,
                num_beams=5,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predicted_answer = extract_answer(decoded_output, input_text)
        predicted_answer = predicted_answer.strip().lower()

        lev_score = Levenshtein.ratio(predicted_answer, ground_truth)
        is_exact = predicted_answer == ground_truth
        is_similar = lev_score >= 0.8

        exact_match += int(is_exact)
        similar_match += int(is_similar)
        total += 1

        predictions.append({
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "levenshtein_score": lev_score,
            "exact_match": is_exact,
            "lenient_match": is_similar
        })

        print(f"[{idx+1}/{len(test_data)}] EM: {exact_match/total:.2%}, Lev≥0.8: {similar_match/total:.2%}")

    print("\n=== Final Evaluation ===")
    print(f"Total Samples             : {total}")
    print(f"Exact Match Accuracy      : {exact_match / total * 100:.2f}%")
    print(f"Levenshtein ≥ 0.8 Accuracy: {similar_match / total * 100:.2f}%")

    # Save predictions
    os.makedirs("llama8bresults", exist_ok=True)
    output_file = "/llama8bresults/predictions_epoch4.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"Predictions saved to {output_file}")

    # Print first 10 examples
    print("\n=== First 10 Predictions ===")
    for i, pred in enumerate(predictions[:10]):
        print(f"\nExample {i+1}")
        print(f"Question        : {pred['question']}")
        print(f"Ground Truth    : {pred['ground_truth']}")
        print(f"Predicted Answer: {pred['predicted_answer']}")
        print(f"Levenshtein     : {pred['levenshtein_score']:.2f}")
        print(f"Exact Match     : {pred['exact_match']}")
        print(f"Lenient Match   : {pred['lenient_match']}")

if __name__ == "__main__":
    main()

"""


Total Samples             : 7175
Exact Match Accuracy      : 53.97%
Levenshtein ≥ 0.8 Accuracy: 56.53%
Predictions saved to /llama8bresults/predictions_epoch3.json

=== First 10 Predictions ===

Example 1
Question        : which country had the most cyclists finish within the top 10?
Ground Truth    : italy
Predicted Answer: spain
Levenshtein     : 0.20
Exact Match     : False
Lenient Match   : False

Example 2
Question        : how many people were murdered in 1940/41?
Ground Truth    : 100,000
Predicted Answer: 100,000
Levenshtein     : 1.00
Exact Match     : True
Lenient Match   : True

Example 3
Question        : how long did it take for the new york americans to win the national cup after 1936?
Ground Truth    : 17 years
Predicted Answer: 2 years
Levenshtein     : 0.80
Exact Match     : False
Lenient Match   : True

Example 4
Question        : alfie's birthday party aired on january 19. what was the airdate of the next episode?
Ground Truth    : january 26, 1995
Predicted Answer: january 26, 1995
Levenshtein     : 1.00
Exact Match     : True
Lenient Match   : True

Example 5
Question        : what is the number of 1st place finishes across all events?
Ground Truth    : 17
Predicted Answer: 12
Levenshtein     : 0.50
Exact Match     : False
Lenient Match   : False

Example 6
Question        : in which competition did hopley finish fist?
Ground Truth    : world junior championships
Predicted Answer: world junior championships
Levenshtein     : 1.00
Exact Match     : True
Lenient Match   : True

Example 7
Question        : what is the total number of films with the language of kannada listed?
Ground Truth    : 15
Predicted Answer: 15
Levenshtein     : 1.00
Exact Match     : True
Lenient Match   : True

Example 8
Question        : what was the number of people attending the toros mexico vs. monterrey flash game?
Ground Truth    : 363
Predicted Answer: 363
Levenshtein     : 1.00
Exact Match     : True
Lenient Match   : True

Example 9
Question        : what time period had no shirt sponsor?
Ground Truth    : 1982-1985
Predicted Answer: 1988-1989
Levenshtein     : 0.78
Exact Match     : False
Lenient Match   : False

Example 10
Question        : when was his first 1st place record?
Ground Truth    : 2000
Predicted Answer: 2000
Levenshtein     : 1.00
Exact Match     : True
Lenient Match   : True
"""
