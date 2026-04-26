# Error Bucket Guide – Lab 7 CineSense

Use this guide when analyzing outputs from Prompt v1 and Prompt v2.

| Error bucket | Meaning | Example |
|---|---|---|
| none | Output is acceptable | Correct sentiment, valid JSON, evidence is present |
| wrong_sentiment | Predicted sentiment differs from gold label | Gold: negative, Predicted: positive |
| missed_positive_aspect | LLM ignores an important positive phrase | Review praises acting, output only discusses pacing |
| missed_negative_aspect | LLM ignores an important negative phrase | Review says "weak pacing", output omits it |
| wrong_aspect_polarity | Aspect found but polarity is wrong | "slow pacing" marked as positive |
| invalid_json | Output is not valid JSON or has extra prose | Markdown text instead of JSON only |
| hallucinated_evidence | Evidence phrase is not present in the review | Adds "great director" when review never says it |
| overconfident | Confidence is high for ambiguous/mixed review | Very mixed review but confidence = high |
| outside_knowledge | Uses movie facts not in the review | Mentions awards, actors, box office |
| other | Important error not covered above | Explain in notes |

Minimum expectation:
- Evaluate at least 20 reviews.
- Compare Prompt v1 and Prompt v2.
- Include at least 3 concrete examples of errors and explain how Prompt v2 attempted to fix them.
