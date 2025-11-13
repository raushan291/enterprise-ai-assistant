import gc
import json
import os

import evaluate
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

from src.config.settings import settings


def test_model(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return pred


def evaluate_model(tokenizer, model, eval_data, verbose=False):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    meteor = evaluate.load("meteor")
    chrf = evaluate.load("chrf")

    predictions = []
    references = []

    print("\n=== EVALUATION STARTED ===\n")

    for sample in eval_data:
        inp = sample["input"]
        ref = sample["target"]

        inputs = tokenizer(inp, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=128)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(pred)
        references.append(ref)

        if verbose:
            print(f"Q: {inp}")
            print(f"Model: {pred}")
            print(f"True : {ref}")
            print("-" * 60)

    # Compute ROUGE-L
    results = rouge.compute(
        predictions=predictions, references=references, use_stemmer=True
    )
    rouge_l = round(results["rougeL"], 4)

    bleu_results = bleu.compute(
        predictions=predictions, references=[[r] for r in references]
    )
    bleu_score = round(bleu_results["bleu"], 4)

    meteor_results = meteor.compute(predictions=predictions, references=references)
    meteor_score = round(meteor_results["meteor"], 4)

    chrf_results = chrf.compute(predictions=predictions, references=references)
    chrf_score = round(chrf_results["score"], 4)

    bert_results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
    )
    bert_f1 = round(sum(bert_results["f1"]) / len(bert_results["f1"]), 4)

    exact_matches = sum(
        p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references)
    )
    exact_match_accuracy = exact_matches / len(predictions)

    return {
        "rougeL_score": rouge_l,
        "bleu_score": bleu_score,
        "meteor_score": meteor_score,
        "chrf_score": chrf_score,
        "bertscore_f1": bert_f1,
        "exact_match_accuracy": exact_match_accuracy,
    }


if __name__ == "__main__":
    # Load evaluation data
    print("\nLoading evaluation data...")
    with open(settings.EVAL_DATA_PATH) as f:
        eval_data = [json.loads(line) for line in f]

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(settings.LORA_PATH)

    results_all = {}

    # Define variants to evaluate sequentially
    variants = [
        ("fp32", dict(dtype=torch.float32)),
        ("fp16", dict(dtype=torch.float16, low_cpu_mem_usage=True)),
        ("int8", dict(dtype="int8")),
    ]

    def cleanup_model(model):
        """Force model + CUDA memory cleanup."""
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Sequential evaluation loop
    for name, opts in variants:
        print("\n\n==============================")
        print(f"Evaluating variant: {name.upper()}")
        print("==============================\n")

        model = None
        try:
            if name == "fp32":
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    settings.BASE_LLM_MODEL
                )
                model = PeftModel.from_pretrained(base_model, settings.LORA_PATH)

            elif name == "fp16":
                try:
                    base_model = AutoModelForSeq2SeqLM.from_pretrained(
                        settings.BASE_LLM_MODEL,
                        torch_dtype=opts["dtype"],
                        low_cpu_mem_usage=True,
                    )
                    model = PeftModel.from_pretrained(base_model, settings.LORA_PATH)
                except Exception as e:
                    print(f"Skipping FP16: {e}")
                    continue

            elif name == "int8":
                model = None
                try:
                    if torch.cuda.is_available() or torch.backends.mps.is_available():
                        print("Using BitsAndBytes quantized model for INT8...")
                        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                        base_model = AutoModelForSeq2SeqLM.from_pretrained(
                            settings.BASE_LLM_MODEL,
                            quantization_config=bnb_config,
                            device_map="auto",
                        )
                        model = PeftModel.from_pretrained(
                            base_model, settings.LORA_PATH
                        )
                    else:
                        print("Using torch.ao.quantization INT8 fallback...")
                        int8_model_file = os.path.join(
                            settings.LORA_PATH + "_int8_cpu", "adapter_model_int8.pt"
                        )
                        if os.path.exists(int8_model_file):
                            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                                settings.BASE_LLM_MODEL
                            )
                            model = PeftModel.from_pretrained(
                                base_model, settings.LORA_PATH
                            )
                            state_dict = torch.load(int8_model_file, map_location="cpu")
                            model.load_state_dict(state_dict, strict=False)
                        else:
                            print("No INT8 fallback file found.")
                            continue
                except Exception as e:
                    print(f"Skipping INT8 evaluation: {e}")
                    continue

            # Evaluate
            if model is not None:
                model.eval()
                metrics = evaluate_model(tokenizer, model, eval_data, verbose=False)

                print(f"\n=== METRICS ({name.upper()}) ===")
                for k, v in metrics.items():
                    print(f"{k}: {v}")
                print("=== DONE ===\n")

                results_all[name] = metrics

                # Test example
                prompt = "What is COVID-19?"
                res = test_model(tokenizer, model, prompt)
                print(f"\n=== Test Example ({name.upper()}) ====")
                print(f"Prompt: {prompt}\nResponse: {res}")

        except Exception as e:
            print(f"Error while evaluating {name.upper()}: {e}")

        finally:
            # Clean up model from memory before loading next model
            cleanup_model(model)

    # Save metrics
    output_path = os.path.join(settings.LORA_PATH, "evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(results_all, f, indent=2)

    print(f"\nAll metrics saved to {output_path}")
