"""
Lab 7 CineSense – Prompt Evaluation Skeleton

This is a scaffold, not a complete starter kit.
Students should complete the TODO sections based on the LLM/API they choose.
"""

import csv
from pathlib import Path

DATA_PATH = Path("data/imdb_sample_50.csv")
PROMPT_PATH = Path("prompts/prompt_template_v1.txt")
OUTPUT_PATH = Path("outputs/result_v1.csv")


def call_llm(prompt: str) -> str:
    """
    TODO:
    Replace this function with your LLM call.
    Options:
    - Gemini API
    - Groq API
    - OpenRouter
    - Ollama local API
    - Manual web UI copy/paste, then skip this script
    """
    raise NotImplementedError("Students should implement the LLM call.")


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    with DATA_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    outputs = []

    for row in rows:
        prompt = prompt_template.replace("{review_text}", row["review_text"])

        # TODO: call your LLM here
        # llm_output = call_llm(prompt)

        llm_output = "TODO: paste or generate LLM output here"

        outputs.append({
            "review_id": row["review_id"],
            "review_text": row["review_text"],
            "gold_sentiment": row["gold_sentiment"],
            "llm_output": llm_output
        })

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["review_id", "review_text", "gold_sentiment", "llm_output"]
        )
        writer.writeheader()
        writer.writerows(outputs)

    print(f"Saved outputs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
