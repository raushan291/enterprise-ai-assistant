import json

import evaluate
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config.settings import settings


def test_model(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return pred


def evaluate_model(tokenizer, model, eval_data):
    rouge = evaluate.load("rouge")

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

        print(f"Q: {inp}")
        print(f"Model: {pred}")
        print(f"True : {ref}")
        print("-" * 60)

    # Compute ROUGE-L
    results = rouge.compute(
        predictions=predictions, references=references, use_stemmer=True
    )
    rouge_l = round(results["rougeL"], 4)

    exact_matches = sum(
        p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references)
    )
    exact_match_accuracy = exact_matches / len(predictions)

    print("\nROUGE-L Score:", rouge_l)
    print("Exact Match Accuracy:", exact_match_accuracy)
    print("\n=== EVALUATION COMPLETE ===\n")

    return {
        "rougeL": rouge_l,
        "exact_match_accuracy": exact_match_accuracy,
        "predictions": predictions,
        "references": references,
    }


if __name__ == "__main__":
    # Load tokenizer and model (base + LoRA)
    tokenizer = AutoTokenizer.from_pretrained(settings.LORA_PATH)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(settings.BASE_LLM_MODEL)
    model = PeftModel.from_pretrained(base_model, settings.LORA_PATH)

    # Load test/eval set (JSONL format expected)
    eval_path = settings.EVAL_DATA_PATH
    eval_data = [json.loads(line) for line in open(eval_path)]

    metrics = evaluate_model(tokenizer, model, eval_data)

    prompt = "What is COVID-19?"
    res = test_model(tokenizer, model, prompt)

    print(f"Prompt: {prompt}\nResponse: {res}")
