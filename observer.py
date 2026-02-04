import json
import time
from huggingface_hub import InferenceClient

OBSERVER_SYSTEM_PROMPT = """
You are a LOGIC AUDITOR.

Evaluate an intermediate reasoning segment.

Detect:
- Invalid or hallucinated premises
- Circular reasoning
- Missing mitigation of obvious constraints
- Ethical oversimplification
- Logical inconsistency

Respond ONLY in valid JSON:

{
  "decision": "PROCEED" or "STOP",
  "confidence": 0.0 to 1.0,
  "failure_type": "premise | circular | mitigation | ethical | inconsistency | none",
  "comment": "short explanation"
}
"""

class ObserverModel:
    """
    Lightweight semantic auditor.
    Makes PROCEED / STOP decisions.
    """

    def __init__(self, model_name, hf_token):
        self.client = InferenceClient(
            model=model_name,
            token=hf_token
        )

    def audit(self, reasoning_segment):
        start = time.time()

        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": OBSERVER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Audit this reasoning:\n\n{reasoning_segment}"}
            ]
        )

        latency = time.time() - start
        raw = response.choices[0].message.content

        try:
            audit_json = json.loads(raw)
        except json.JSONDecodeError:
            audit_json = {
                "decision": "STOP",
                "confidence": 0.0,
                "failure_type": "inconsistency",
                "comment": "Invalid JSON from Observer"
            }

        return audit_json, latency
