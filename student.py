# student.py
# Student Model Interface for Reasoning Generation
# UPDATED VERSION - with working model alternatives

import os
import json
import time
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Try importing huggingface_hub
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Run: pip install huggingface_hub")

from config import ModelConfig, STUDENT_SYSTEM_PROMPT, STUDENT_USER_TEMPLATE


@dataclass
class StudentResponse:
    """Response from the Student model."""
    full_response: str
    reasoning_segment: str  # Truncated at OIP
    tokens_used: int
    generation_time: float
    model: str
    truncated_at_oip: bool


class StudentModel:
    """
    Student Model: High-capability reasoning model.
    
    Generates step-by-step reasoning for tasks.
    
    NOTE: DeepSeek-R1 models may not work well with HuggingFace Inference API.
    Recommended alternatives that work reliably:
    - Qwen/Qwen2.5-7B-Instruct (recommended)
    - mistralai/Mistral-7B-Instruct-v0.3
    - microsoft/Phi-3-mini-4k-instruct
    - meta-llama/Llama-3.1-8B-Instruct
    """
    
    # Models known to work well with HF Inference API
    WORKING_MODELS = [
        "Qwen/Qwen2.5-7B-Instruct",      # Recommended - good reasoning
        "Qwen/Qwen2.5-14B-Instruct",     # Better but slower
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.1-8B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
    ]
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.client = None
        
        # Check if configured model might have issues
        if "deepseek" in self.config.student_model.lower():
            print(f"⚠ Warning: {self.config.student_model} may not work with HF Inference API")
            print(f"  Consider using: {self.WORKING_MODELS[0]}")
        
        if HF_AVAILABLE and self.config.hf_api_token:
            self.client = InferenceClient(token=self.config.hf_api_token)
        elif HF_AVAILABLE:
            # Try without token (for public models)
            self.client = InferenceClient()
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (approx 4 chars per token)."""
        return len(text) // 4
    
    def _extract_reasoning(self, response: str, oip: int) -> tuple:
        """
        Extract reasoning segment up to OIP tokens.
        
        Returns:
            (reasoning_segment, was_truncated)
        """
        # Remove common thinking tags if present
        cleaned = re.sub(r'<think>|</think>|<reasoning>|</reasoning>|<\|begin_of_thought\|>|<\|end_of_thought\|>', '', response)
        cleaned = cleaned.strip()
        
        # Estimate token count and truncate if needed
        estimated_tokens = self._estimate_tokens(cleaned)
        
        if estimated_tokens <= oip:
            return cleaned, False
        
        # Truncate to approximately OIP tokens
        char_limit = oip * 4  # Rough conversion
        truncated = cleaned[:char_limit]
        
        # Try to truncate at sentence boundary
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        cut_point = max(last_period, last_newline)
        
        if cut_point > char_limit * 0.7:  # Only use if we keep most content
            truncated = truncated[:cut_point + 1]
        
        return truncated.strip(), True
    
    def generate(self, task: str, oip: int = 150) -> StudentResponse:
        """
        Generate reasoning for a task.
        
        Args:
            task: The task/prompt to reason about
            oip: Optimal Intervention Point (max tokens for reasoning segment)
        
        Returns:
            StudentResponse with reasoning
        """
        start_time = time.time()
        
        user_prompt = STUDENT_USER_TEMPLATE.format(task=task)
        
        if self.client is None:
            # Fallback: return mock response for testing
            return self._mock_generate(task, oip, start_time)
        
        # Try primary model first, then fallbacks
        models_to_try = [self.config.student_model] + self.WORKING_MODELS[:2]
        
        for model in models_to_try:
            try:
                response = self.client.chat_completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=self.config.max_tokens_student,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                
                full_response = response.choices[0].message.content
                
                # Check if response is empty (model issue)
                if not full_response or len(full_response.strip()) < 10:
                    print(f"  ⚠ Empty response from {model}, trying fallback...")
                    continue
                
                generation_time = time.time() - start_time
                
                reasoning_segment, truncated = self._extract_reasoning(full_response, oip)
                
                return StudentResponse(
                    full_response=full_response,
                    reasoning_segment=reasoning_segment,
                    tokens_used=self._estimate_tokens(full_response),
                    generation_time=generation_time,
                    model=model,
                    truncated_at_oip=truncated
                )
                
            except Exception as e:
                print(f"  ⚠ Error with {model}: {str(e)[:50]}...")
                continue
        
        # All models failed - return mock
        print(f"  ✗ All models failed, using mock response")
        return self._mock_generate(task, oip, start_time, error="All models failed")
    
    def _mock_generate(self, task: str, oip: int, start_time: float, error: str = None) -> StudentResponse:
        """Generate mock response for testing without API."""
        
        mock_reasoning = f"""Let me think through this step by step.

First, I need to understand what's being asked: {task[:100]}...

Analyzing the key components:
1. The main challenge here is understanding the underlying assumptions
2. I should consider multiple perspectives
3. Let me work through the logic carefully

Based on my analysis, I believe the approach should be...

[This is a mock response for testing purposes]
"""
        
        if error:
            mock_reasoning += f"\n[API Error: {error}]"
        
        reasoning_segment, truncated = self._extract_reasoning(mock_reasoning, oip)
        
        return StudentResponse(
            full_response=mock_reasoning,
            reasoning_segment=reasoning_segment,
            tokens_used=self._estimate_tokens(mock_reasoning),
            generation_time=time.time() - start_time,
            model="mock-model (no API)",
            truncated_at_oip=truncated
        )


if __name__ == "__main__":
    # Test the student model
    print("Testing Student Model...")
    print("=" * 60)
    
    student = StudentModel()
    
    test_task = "If 2 shirts take 2 hours to dry, how long will 20 shirts take?"
    
    response = student.generate(test_task, oip=150)
    
    print(f"Task: {test_task}")
    print(f"\nModel: {response.model}")
    print(f"Tokens used: {response.tokens_used}")
    print(f"Generation time: {response.generation_time:.2f}s")
    print(f"Truncated at OIP: {response.truncated_at_oip}")
    print(f"\nReasoning Segment:\n{response.reasoning_segment[:500]}...")
