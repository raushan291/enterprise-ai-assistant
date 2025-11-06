import json
import os
import re
from typing import Optional, Union

from pydantic import BaseModel, Field, ValidationError

from src.rag_pipeline.orchestrator import rag_response


class QAPair(BaseModel):
    """Represents a validated question-answer pair."""

    question: str = Field(..., min_length=3, description="A natural language question.")
    answer: str = Field(..., min_length=3, description="The corresponding answer text.")


def extract_json(text: str) -> Optional[Union[list, dict]]:
    """Extract JSON array/object from raw LLM output."""
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if match:
        json_str = match.group(1)
        json_str = json_str.replace("\n", " ").replace(",]", "]").replace(",}", "}")
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None


def parse_qa_pairs(response: str) -> list[QAPair]:
    """Parse a raw LLM response string into a list of validated QAPair objects."""
    data = extract_json(response)
    if not data:
        print("Could not extract valid JSON, using fallback.")
        return [QAPair(question="What is this about?", answer=response[:200])]

    try:
        # If it's a single dict instead of a list
        if isinstance(data, dict):
            data = [data]
        return [QAPair(**item) for item in data]
    except ValidationError as e:
        print("Validation failed:", e)
        return [QAPair(question="What is this about?", answer=response[:200])]


def generate_qa_pairs(chunks_file: str, output_file: str) -> list[QAPair]:
    """Generate Q/A pairs for each text chunk in a file using an LLM via `rag_response`."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    qa_pairs: list[QAPair] = []

    with open(chunks_file, "r") as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            prompt = f"""
                Generate 3 question-answer pairs for the text below.
                Return only valid JSON - no markdown, no other formats.
                Format:
                [
                {{"question": "...", "answer": "..."}},
                ...
                ]

                Text:
                {text}
                """

            response = rag_response(prompt)["answer"]
            pairs = parse_qa_pairs(response)

            qa_pairs.extend(pairs)

    with open(output_file, "w") as f:
        json.dump([qa.model_dump() for qa in qa_pairs], f, indent=2)

    print(f"Generated {len(qa_pairs)} Q/A pairs to {output_file}")
    return qa_pairs
