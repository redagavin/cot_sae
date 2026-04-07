# ABOUTME: Runs all 50 questions under hint conditions with activation caching.
# ABOUTME: Deduplicates no-hint runs (format-independent) to avoid redundant computation.

import json
import torch
from tqdm import tqdm

from src.config import (
    DATA_DIR,
    OUTPUTS_DIR,
    HINT_FORMATS,
    ANSWER_LETTERS,
)
from src.data import build_prompt, format_for_model, insert_hint, parse_answer, check_mentions_hint
from src.generate import load_model, generate_with_cache, pick_false_answer


def main():
    activations_dir = OUTPUTS_DIR / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = OUTPUTS_DIR / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "selected_questions.json") as f:
        questions = json.load(f)

    print("Loading model...")
    model = load_model()
    tokenizer = model.tokenizer

    all_metadata = []

    print(f"Running {len(questions)} questions...")
    for q_idx, q in enumerate(tqdm(questions, desc="Questions")):
        correct_idx = q["correct_answer"]
        false_idx = pick_false_answer(correct_idx, seed=q_idx)

        # Generate no-hint ONCE per question (format-independent)
        no_hint_msg = build_prompt(
            question=q["question"], choices=q["choices"], hint_text=""
        )
        no_hint_formatted = format_for_model(tokenizer, no_hint_msg)
        no_hint_text, no_hint_acts = generate_with_cache(model, no_hint_formatted)
        no_hint_predicted = parse_answer(no_hint_text)
        no_hint_prompt_len = len(model.to_tokens(no_hint_formatted, prepend_bos=False)[0])

        no_hint_run_id = f"q{q_idx:03d}_no_hint"
        torch.save(no_hint_acts, activations_dir / f"{no_hint_run_id}.pt")

        no_hint_entry = {
            "run_id": no_hint_run_id,
            "question_idx": q_idx,
            "hint_format": "none",
            "condition": "no_hint",
            "correct_answer": correct_idx,
            "false_answer": false_idx,
            "predicted": no_hint_predicted,
            "response": no_hint_text,
            "hint_following": False,
            "mentions_hint": False,
            "prompt_length": no_hint_prompt_len,
        }
        all_metadata.append(no_hint_entry)

        # Generate true-hint and false-hint for each format
        for hint_format in HINT_FORMATS:
            for condition in ["true_hint", "false_hint"]:
                hint_text = insert_hint(
                    condition=condition,
                    hint_format=hint_format,
                    correct_idx=correct_idx,
                    false_answer_idx=false_idx,
                )
                user_msg = build_prompt(
                    question=q["question"], choices=q["choices"], hint_text=hint_text
                )
                formatted = format_for_model(tokenizer, user_msg)
                text, activations = generate_with_cache(model, formatted)
                predicted = parse_answer(text)
                prompt_len = len(model.to_tokens(formatted, prepend_bos=False)[0])

                run_id = f"q{q_idx:03d}_{hint_format}_{condition}"
                torch.save(activations, activations_dir / f"{run_id}.pt")

                entry = {
                    "run_id": run_id,
                    "question_idx": q_idx,
                    "hint_format": hint_format,
                    "condition": condition,
                    "correct_answer": correct_idx,
                    "false_answer": false_idx,
                    "predicted": predicted,
                    "response": text,
                    "hint_following": (
                        condition == "false_hint" and predicted == false_idx
                    ),
                    "mentions_hint": check_mentions_hint(text, hint_format),
                    "prompt_length": prompt_len,
                }
                all_metadata.append(entry)

    with open(metadata_dir / "generation_metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2)

    hint_following = sum(1 for m in all_metadata if m["hint_following"])
    total_false = sum(1 for m in all_metadata if m["condition"] == "false_hint")
    print(f"Generation complete. Hint-following rate: {hint_following}/{total_false}")
    # Total runs: 50 no-hint + 50*3*2 hint runs = 350 forward passes


if __name__ == "__main__":
    main()
