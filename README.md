# Catching Reasoning Before It Derails

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2026.XXXXX-b31b1b.svg)](https://arxiv.org/)

> **Inference-Time Process Supervision for Large Language Models**

A dual-model framework that monitors LLM reasoning in real-time and intervenes when semantic failures are detected—before errors cascade into confident hallucinations.

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Precision** | 84% |
| **Recall** | 70% |
| **F1 Score** | 0.76 |
| **Token Savings** | 44.2% |

| Task Category | Detection Rate |
|---------------|----------------|
| Logic Traps | **100%** ✓ |
| Ethical Dilemmas | **80%** ✓ |
| Business Strategy | 60% (pass-through) |
| Logical Paradoxes | 30% |

---

## 📖 Abstract

Large Language Models increasingly rely on extended inference-time computation. However, **longer reasoning doesn't guarantee correctness**—models often follow flawed premises, ethical oversimplifications, or self-referential loops.

We propose a **Dual-Model Process Supervision Framework**:
- A **Student** model generates reasoning
- An **Observer** model evaluates intermediate steps at a fixed checkpoint (OIP)
- Flawed trajectories are **halted early**, saving compute and preventing hallucinations

---

## 🏗️ Architecture

```
┌─────────────────┐
│   User Prompt   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Student Model  │  (Qwen2.5-7B-Instruct)
│  Generate       │
│  Reasoning      │
└────────┬────────┘
         │
         ▼ (at OIP = 150 tokens)
┌─────────────────┐
│ Observer Model  │  (Llama-3.1-8B-Instruct)
│ Evaluate for:   │
│ • Invalid premise│
│ • Circular logic │
│ • Ethics issues  │
│ • Inconsistency  │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│PROCEED│ │ STOP  │
│Continue│ │ Halt  │
└───────┘ └───────┘
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/pranav-k-27/catching-reasoning-before-it-derails.git
cd catching-reasoning-before-it-derails
pip install -r requirements.txt
```

### Setup

```bash
cp .env.example .env
# Add your HuggingFace API token to .env
```

### Run Experiments

```bash
# Test run (mock mode, no API needed)
python run.py --test

# Full experiments (40 prompts)
python run.py

# OIP Ablation study
python run.py --ablation

# Analyze results
python run.py --analyze

# Generate plots
python figures/generate_plots.py
```

---

## 📁 Project Structure

```
catching-reasoning-before-it-derails/
├── run.py                    # Main entry point
├── config.py                 # Model & experiment configuration
├── prompts.py                # 40 evaluation prompts (10 per category)
├── student.py                # Student model interface
├── observer.py               # Observer model interface
├── analyze_results.py        # Generate paper tables from results
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
│
├── experiments/
│   ├── run_experiments.py    # ExperimentRunner class
│   └── run_ablation.py       # Ablation study runner
│
├── figures/
│   ├── generate_plots.py     # Generate publication figures
│   ├── token_comparison.png  # Token usage comparison
│   └── ablation_plot.png     # OIP ablation results
│
├── results/                  # Experiment outputs (JSON)
│   ├── results_*.json
│   └── ablation_results_*.json
│
└── logs/                     # Execution logs
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
@dataclass
class ModelConfig:
    # Models
    student_model: str = "Qwen/Qwen2.5-7B-Instruct"
    observer_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Decoding
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens_student: int = 512
    max_tokens_observer: int = 256

@dataclass  
class ExperimentConfig:
    default_oip: int = 150  # Optimal Intervention Point
    oip_ablation_values: tuple = (100, 150, 200)
```

---

## 📊 Results

### Confusion Matrix

```
                    Predicted STOP    Predicted PROCEED
Actual STOP              21                  9
(flawed reasoning)       (TP)               (FN)

Actual PROCEED           4                   6
(valid reasoning)        (FP)               (TN)
```

### Token Efficiency

| Category | Student-only | Supervised | Savings |
|----------|-------------|------------|---------|
| Business Strategy | 650 | 450 | 31% |
| Logic Traps | 265 | 150 | 43% |
| Ethical Dilemmas | 533 | 145 | **73%** |
| Logical Paradoxes | 123 | 131 | -6% |
| **Overall** | **393** | **219** | **44.2%** |

### Example: Logic Trap Detection

**Task:** *"If 2 shirts take 2 hours to dry, how long will 20 shirts take?"*

| Setting | Result | Tokens |
|---------|--------|--------|
| Student-only | ❌ "20 hours" (wrong) | ~500 |
| Process-supervised | ✅ STOP (premise error) | ~150 |

---

## 🔬 Task Categories

| Category | Description | Expected Behavior |
|----------|-------------|-------------------|
| **Business Strategy** | Open-ended, exploratory reasoning | PROCEED |
| **Logic Traps** | Misleading premises (e.g., linear scaling) | STOP |
| **Ethical Dilemmas** | Morally complex decisions | STOP |
| **Logical Paradoxes** | Self-referential statements | STOP |

---

## 📝 Observer Failure Types

The Observer classifies reasoning failures into 5 types:

| Type | Description | Example |
|------|-------------|---------|
| `premise` | Invalid assumptions | "More shirts = more time" |
| `circular` | Self-referential loops | Liar paradox |
| `mitigation` | Ignoring constraints | Missing risk analysis |
| `ethical` | Moral oversimplification | Utilitarian reduction |
| `inconsistency` | Logical contradictions | A and not-A |

---

## 🧪 Running Your Own Experiments

### Add Custom Prompts

Edit `prompts.py`:

```python
PROMPTS = [
    {
        "id": "custom_01",
        "category": "logic_trap",
        "task": "Your custom prompt here",
        "expected_behavior": "stop"  # or "proceed"
    },
    # ...
]
```

### Use Different Models

```python
# In config.py
student_model: str = "your-preferred/model"
observer_model: str = "your-observer/model"
```

---

## 📄 Citation

```bibtex
@article{vachharajani2026catching,
  title={Catching Reasoning Before It Derails: Inference-Time Process Supervision for Large Language Models},
  author={Vachharajani, Pranav},
  journal={arXiv preprint arXiv:2026.XXXXX},
  year={2026}
}
```

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for model hosting
- [Qwen](https://github.com/QwenLM/Qwen) and [Meta Llama](https://llama.meta.com/) teams

---

## 📬 Contact

**Pranav V**  


---

<p align="center">
  <i>Process supervision: Because thinking longer isn't always thinking better.</i>
</p>
