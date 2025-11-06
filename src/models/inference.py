import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config.settings import settings

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = settings.OPENAI_API_KEY
except ImportError:
    OPENAI_AVAILABLE = False

# Model setup
print("Loading local Hugging Face model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(settings.LOCAL_LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.LOCAL_LLM_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    LOCAL_MODEL_READY = True
    print(f"Local model loaded on {device}.")
except Exception as e:
    print(f"Failed to load local model: {e}")
    LOCAL_MODEL_READY = False


def generate_response(prompt: str) -> str:
    """Generate response using local HF model (default) with fallback to OpenAI."""
    # Try local model first
    if LOCAL_MODEL_READY:
        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                early_stopping=True,
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            print(f"Local generation failed: {e}")

    # Fallback to OpenAI
    if OPENAI_AVAILABLE:
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model=settings.FALLBACK_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI fallback failed: {e}")

    # If both fail
    return "Unable to generate response, no model available."
