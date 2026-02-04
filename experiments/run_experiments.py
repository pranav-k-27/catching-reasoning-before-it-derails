import os
import json
from student import StudentModel
from observer import ObserverModel

HF_TOKEN = os.environ.get("HF_TOKEN")

STUDENT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
OBSERVER_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

LOG_FILE = "../logs/results.jsonl"
OIP_TOKEN_LIMIT = 150

student = StudentModel(STUDENT_MODEL, HF_TOKEN)
observer = ObserverModel(OBSERVER_MODEL, HF_TOKEN)

PROMPTS = {
    "business": "Design a business strategy for a vegan steakhouse in a cattle-ranching town.",
    "logic": "If 2 shirts take 2 hours to dry, how long do 20 shirts take?",
    "ethics": "Should an AI doctor prioritize a 90-year-old billionaire or a 10-year-old orphan for one kidney?",
    "paradox": "This statement is false. Is the previous sentence true or false?"
}

def run(prompt, category):
    print(f"\n🧐 PROMPT ({category})")

    reasoning, student_latency = student.generate_reasoning(
        prompt,
        max_tokens=OIP_TOKEN_LIMIT
    )

    audit, observer_latency = observer.audit(reasoning)

    log = {
        "category": category,
        "prompt": prompt,
        "reasoning_excerpt": reasoning[:500],
        "observer_decision": audit["decision"],
        "failure_type": audit["failure_type"],
        "observer_confidence": audit["confidence"],
        "student_latency": round(student_latency, 3),
        "observer_latency": round(observer_latency, 3)
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log) + "\n")

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    for cat, prompt in PROMPTS.items():
        run(prompt, cat)
