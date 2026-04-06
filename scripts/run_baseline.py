# ABOUTME: Runs 200 MMLU questions through Gemma 2 2B-it with no hint to establish baseline.
# ABOUTME: Filters to 50 correctly answered questions and saves them for the full experiment.

import json
import random
from tqdm import tqdm

from src.config import DATA_DIR, BASELINE_POOL_SIZE, TARGET_CORRECT
from src.data import load_mmlu, build_prompt, format_for_model, parse_answer
from src.generate import load_model, generate_response


def main():
    random.seed(42)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {BASELINE_POOL_SIZE} MMLU questions...")
    questions = load_mmlu(BASELINE_POOL_SIZE, seed=42)

    print("Loading model...")
    model = load_model()
    tokenizer = model.tokenizer

    correct = []
    results = []

    print("Running baseline (no hint)...")
    for i, q in enumerate(tqdm(questions)):
        user_msg = build_prompt(
            question=q["question"],
            choices=q["choices"],
            hint_text="",
        )
        formatted = format_for_model(tokenizer, user_msg)
        response = generate_response(model, formatted)
        predicted = parse_answer(response)
        is_correct = predicted == q["answer"]

        result = {
            "index": i,
            "question": q["question"],
            "choices": q["choices"],
            "correct_answer": q["answer"],
            "subject": q.get("subject", "unknown"),
            "response": response,
            "predicted": predicted,
            "is_correct": is_correct,
        }
        results.append(result)

        if is_correct:
            correct.append(result)

        if len(correct) >= TARGET_CORRECT:
            print(f"Found {TARGET_CORRECT} correct answers after {i + 1} questions.")
            break

    with open(DATA_DIR / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    selected = correct[:TARGET_CORRECT]
    with open(DATA_DIR / "selected_questions.json", "w") as f:
        json.dump(selected, f, indent=2)

    print(f"Baseline complete: {len(correct)}/{len(results)} correct.")
    if len(selected) < TARGET_CORRECT:
        print(f"WARNING: Only found {len(selected)}/{TARGET_CORRECT} correct answers. "
              f"Consider increasing BASELINE_POOL_SIZE or checking prompt formatting.")
    print(f"Selected {len(selected)} questions saved to {DATA_DIR / 'selected_questions.json'}")


if __name__ == "__main__":
    main()
