# scripts/run_divergence_generation.py
# ABOUTME: Phase 1 of divergence localization — generates CoT under all conditions.
# ABOUTME: HuggingFace batched generation with forward hooks for SAE encoding at fractional positions.

import json
import os
import torch
from tqdm import tqdm
from datasets import load_dataset

from src.config import (
    MMLU_DATASET, MMLU_SPLIT, HINT_FORMATS, MAX_NEW_TOKENS,
    SELECTED_LAYERS, SAE_WIDTHS, N_FRACTIONS, DIVERGENCE_DIR, BATCH_SIZE,
)
from src.data import (
    build_prompt, format_for_model, insert_hint, parse_answer, check_mentions_hint,
)
from src.generate import pick_false_answer
from src.sae_analysis import load_sae
from src.hf_model import (
    load_hf_model, tokenize_batch, extract_generation_lengths,
    register_layer_hooks, remove_hooks,
)
from src.fractional import compute_fraction_indices
from src.fractional_sae import encode_at_fractions


def split_into_chunks(items: list, n_chunks: int) -> list[list]:
    """Split a list into n roughly equal chunks."""
    k, m = divmod(len(items), n_chunks)
    return [items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]


def build_run_metadata(
    run_id, question_idx, hint_format, condition,
    correct_answer, false_answer, predicted, response,
    prompt_length, gen_length,
) -> dict:
    """Assemble metadata for a single generation run."""
    return {
        "run_id": run_id,
        "question_idx": question_idx,
        "hint_format": hint_format,
        "condition": condition,
        "correct_answer": correct_answer,
        "false_answer": false_answer,
        "predicted": predicted,
        "response": response,
        "hint_following": (condition == "false_hint" and predicted == false_answer),
        "mentions_hint": check_mentions_hint(response, hint_format) if hint_format != "none" else False,
        "prompt_length": prompt_length,
        "gen_length": gen_length,
    }


def generate_batch(model, tokenizer, prompts, batch_size):
    """Batched text generation without activation caching.

    Returns (output_tokens, padded_prompt_length, prompt_attention_mask).
    """
    encoded = tokenize_batch(tokenizer, prompts)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    padded_prompt_length = input_ids.shape[1]

    with torch.no_grad():
        output_tokens = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            do_sample=False,
        )

    return output_tokens, padded_prompt_length, attention_mask


def forward_with_hooks(model, input_ids, attention_mask, captured):
    """Run forward pass with hooks already registered. Populates captured dict."""
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)


def process_batch_with_sae(
    model, tokenizer, saes, prompts_info, batch_size,
):
    """Generate text, run hooked forward pass, encode SAE features at fractional positions."""
    results = []

    for batch_start in range(0, len(prompts_info), batch_size):
        batch = prompts_info[batch_start:batch_start + batch_size]
        prompts = [info["formatted_prompt"] for info in batch]

        output_tokens, padded_prompt_length, prompt_attention_mask = generate_batch(
            model, tokenizer, prompts, batch_size
        )

        eos_token_id = tokenizer.eos_token_id or 1
        gen_lengths = extract_generation_lengths(
            output_tokens, padded_prompt_length, eos_token_id
        )

        captured = {}
        hooks = register_layer_hooks(model, SELECTED_LAYERS, captured)

        full_attention_mask = torch.ones_like(output_tokens)
        full_attention_mask[:, :padded_prompt_length] = prompt_attention_mask
        for i in range(len(batch)):
            eos_abs = padded_prompt_length + gen_lengths[i]
            if eos_abs < output_tokens.shape[1]:
                full_attention_mask[i, eos_abs:] = 0

        forward_with_hooks(model, output_tokens, full_attention_mask, captured)
        remove_hooks(hooks)

        for i, info in enumerate(batch):
            gen_len = max(gen_lengths[i], 1)

            gen_start = padded_prompt_length
            gen_tokens = output_tokens[i, gen_start:gen_start + gen_len]
            response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            predicted = parse_answer(response)

            fraction_indices = compute_fraction_indices(
                gen_length=gen_len, n_fractions=N_FRACTIONS,
                prompt_length=padded_prompt_length,
            )

            sae_features = {}
            for layer in SELECTED_LAYERS:
                residual = captured[layer][i]
                for width_k in SAE_WIDTHS:
                    sae = saes[(layer, width_k)]
                    sparse_list = encode_at_fractions(sae, residual, fraction_indices)
                    sae_features[f"L{layer}_W{width_k}k"] = sparse_list

            meta = build_run_metadata(
                run_id=info["run_id"],
                question_idx=info["question_idx"],
                hint_format=info.get("hint_format", "none"),
                condition=info["condition"],
                correct_answer=info["correct_answer"],
                false_answer=info["false_answer"],
                predicted=predicted,
                response=response,
                prompt_length=padded_prompt_length,
                gen_length=gen_len,
            )
            results.append({"metadata": meta, "sae_features": sae_features})

        del captured
        torch.cuda.empty_cache()

    return results


def main():
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    n_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))

    features_dir = DIVERGENCE_DIR / "features"
    metadata_dir = DIVERGENCE_DIR / "metadata"
    features_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print(f"Task {task_id}/{n_tasks}: Loading MMLU...")
    ds = load_dataset(MMLU_DATASET, MMLU_SPLIT, split="test")
    all_questions = [ds[i] for i in range(len(ds))]
    chunks = split_into_chunks(all_questions, n_tasks)
    my_questions = chunks[task_id]
    offset = sum(len(chunks[i]) for i in range(task_id))
    print(f"Task {task_id}: questions {offset} to {offset + len(my_questions) - 1}")

    print("Loading model...")
    model, tokenizer = load_hf_model()

    print("Running baseline to find correctly-answered questions...")
    correct_questions = []

    for batch_start in tqdm(range(0, len(my_questions), BATCH_SIZE), desc="Baseline"):
        batch_qs = my_questions[batch_start:batch_start + BATCH_SIZE]
        prompts = []
        for local_idx, q in enumerate(batch_qs):
            global_idx = offset + batch_start + local_idx
            user_msg = build_prompt(question=q["question"], choices=q["choices"], hint_text="")
            prompts.append(format_for_model(tokenizer, user_msg))

        output_tokens, padded_prompt_length, _ = generate_batch(
            model, tokenizer, prompts, BATCH_SIZE
        )
        eos_token_id = tokenizer.eos_token_id or 1
        gen_lengths = extract_generation_lengths(
            output_tokens, padded_prompt_length, eos_token_id
        )

        for i, q in enumerate(batch_qs):
            global_idx = offset + batch_start + i
            gen_len = max(gen_lengths[i], 1)
            gen_tokens = output_tokens[i, padded_prompt_length:padded_prompt_length + gen_len]
            response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            predicted = parse_answer(response)

            if predicted == q["answer"]:
                correct_questions.append({
                    "global_idx": global_idx,
                    "question": q,
                    "false_answer": pick_false_answer(q["answer"], seed=global_idx),
                    "baseline_response": response,
                })

    print(f"Task {task_id}: {len(correct_questions)}/{len(my_questions)} correct")

    print("Loading SAEs...")
    saes = {}
    for layer in SELECTED_LAYERS:
        for width_k in SAE_WIDTHS:
            saes[(layer, width_k)] = load_sae(layer, width_k)

    print("Running full generation with SAE encoding...")
    all_prompts = []
    for cq in correct_questions:
        q = cq["question"]
        global_idx = cq["global_idx"]
        correct_idx = q["answer"]
        false_idx = cq["false_answer"]

        user_msg = build_prompt(question=q["question"], choices=q["choices"], hint_text="")
        all_prompts.append({
            "formatted_prompt": format_for_model(tokenizer, user_msg),
            "question_idx": global_idx,
            "condition": "no_hint",
            "hint_format": "none",
            "correct_answer": correct_idx,
            "false_answer": false_idx,
            "run_id": f"q{global_idx:05d}_no_hint",
        })

        for hint_format in HINT_FORMATS:
            for condition in ["true_hint", "false_hint"]:
                hint_text = insert_hint(
                    condition=condition, hint_format=hint_format,
                    correct_idx=correct_idx, false_answer_idx=false_idx,
                )
                user_msg = build_prompt(
                    question=q["question"], choices=q["choices"], hint_text=hint_text,
                )
                all_prompts.append({
                    "formatted_prompt": format_for_model(tokenizer, user_msg),
                    "question_idx": global_idx,
                    "condition": condition,
                    "hint_format": hint_format,
                    "correct_answer": correct_idx,
                    "false_answer": false_idx,
                    "run_id": f"q{global_idx:05d}_{hint_format}_{condition}",
                })

    all_results = process_batch_with_sae(model, tokenizer, saes, all_prompts, BATCH_SIZE)

    print(f"Saving {len(all_results)} results...")
    all_metadata = [r["metadata"] for r in all_results]
    with open(metadata_dir / f"metadata_task{task_id}.json", "w") as f:
        json.dump(all_metadata, f, indent=2)

    for result in all_results:
        run_id = result["metadata"]["run_id"]
        torch.save(result["sae_features"], features_dir / f"{run_id}.pt")

    print(f"Task {task_id} complete: {len(correct_questions)} correct, {len(all_results)} runs saved")


if __name__ == "__main__":
    main()
