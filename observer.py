# observer.py
# Observer Model Interface for Reasoning Evaluation

import os
import json
import time
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Try importing huggingface_hub
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Run: pip install huggingface_hub")

from config import ModelConfig, OBSERVER_SYSTEM_PROMPT, OBSERVER_USER_TEMPLATE


class FailureType(Enum):
    """Types of reasoning failures detected by Observer."""
    NONE = "none"
    PREMISE = "premise"
    CIRCULAR = "circular"
    MITIGATION = "mitigation"
    ETHICAL = "ethical"
    INCONSISTENCY = "inconsistency"


@dataclass
class ObserverDecision:
    """Decision from the Observer model."""
    decision: str  # "PROCEED" or "STOP"
    confidence: float
    failure_type: str
    comment: str
    raw_response: str
    evaluation_time: float
    model: str
    parse_success: bool


class ObserverModel:
    """
    Observer Model: Lightweight reasoning auditor.
    
    Evaluates intermediate reasoning for semantic failures.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.client = None
        
        if HF_AVAILABLE and self.config.hf_api_token:
            self.client = InferenceClient(token=self.config.hf_api_token)
        elif HF_AVAILABLE:
            self.client = InferenceClient()
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from model response, handling common issues."""
        
        # Try to extract JSON from response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        response = response.strip()
        
        # Try direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Return default on parse failure
        return None
    
    def evaluate(self, reasoning: str, task: str) -> ObserverDecision:
        """
        Evaluate a reasoning segment for semantic failures.
        
        Args:
            reasoning: The reasoning segment from Student model
            task: Original task for context
        
        Returns:
            ObserverDecision with evaluation results
        """
        start_time = time.time()
        
        user_prompt = OBSERVER_USER_TEMPLATE.format(
            reasoning=reasoning,
            task=task
        )
        
        if self.client is None:
            return self._mock_evaluate(reasoning, task, start_time)
        
        try:
            response = self.client.chat_completion(
                model=self.config.observer_model,
                messages=[
                    {"role": "system", "content": OBSERVER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.config.max_tokens_observer,
                temperature=0.3,  # Lower temperature for more consistent evaluation
            )
            
            raw_response = response.choices[0].message.content
            evaluation_time = time.time() - start_time
            
            parsed = self._parse_json_response(raw_response)
            
            if parsed:
                return ObserverDecision(
                    decision=parsed.get("decision", "PROCEED").upper(),
                    confidence=float(parsed.get("confidence", 0.5)),
                    failure_type=parsed.get("failure_type", "none"),
                    comment=parsed.get("comment", ""),
                    raw_response=raw_response,
                    evaluation_time=evaluation_time,
                    model=self.config.observer_model,
                    parse_success=True
                )
            else:
                # Parse failed - default to PROCEED with low confidence
                return ObserverDecision(
                    decision="PROCEED",
                    confidence=0.3,
                    failure_type="none",
                    comment="Failed to parse observer response",
                    raw_response=raw_response,
                    evaluation_time=evaluation_time,
                    model=self.config.observer_model,
                    parse_success=False
                )
                
        except Exception as e:
            print(f"Error evaluating reasoning: {e}")
            return self._mock_evaluate(reasoning, task, start_time, error=str(e))
    
    def _mock_evaluate(self, reasoning: str, task: str, start_time: float, error: str = None) -> ObserverDecision:
        """Generate mock evaluation for testing without API."""
        
        # Simple heuristic-based mock evaluation
        reasoning_lower = reasoning.lower()
        task_lower = task.lower()
        
        # Check for logic trap indicators
        if any(x in task_lower for x in ["shirts", "dry", "machines", "widgets", "pills"]):
            if any(x in reasoning_lower for x in ["multiply", "times", "20 hours", "100 minutes", "linear"]):
                return ObserverDecision(
                    decision="STOP",
                    confidence=0.85,
                    failure_type="premise",
                    comment="[MOCK] Invalid linear scaling assumption detected",
                    raw_response="mock response",
                    evaluation_time=time.time() - start_time,
                    model="mock-observer",
                    parse_success=True
                )
        
        # Check for paradox indicators
        if any(x in task_lower for x in ["paradox", "false", "liar", "barber", "omnipotent"]):
            return ObserverDecision(
                decision="STOP",
                confidence=0.90,
                failure_type="circular",
                comment="[MOCK] Self-referential paradox detected",
                raw_response="mock response",
                evaluation_time=time.time() - start_time,
                model="mock-observer",
                parse_success=True
            )
        
        # Check for ethical oversimplification
        if any(x in task_lower for x in ["should", "ethical", "prioritize", "choose between"]):
            if any(x in reasoning_lower for x in ["clearly", "obviously", "the answer is", "must"]):
                return ObserverDecision(
                    decision="STOP",
                    confidence=0.80,
                    failure_type="ethical",
                    comment="[MOCK] Ethical oversimplification detected",
                    raw_response="mock response",
                    evaluation_time=time.time() - start_time,
                    model="mock-observer",
                    parse_success=True
                )
        
        # Default: PROCEED for business/benign tasks
        return ObserverDecision(
            decision="PROCEED",
            confidence=0.75,
            failure_type="none",
            comment="[MOCK] Reasoning appears valid" + (f" [Error: {error}]" if error else ""),
            raw_response="mock response",
            evaluation_time=time.time() - start_time,
            model="mock-observer",
            parse_success=True
        )


if __name__ == "__main__":
    # Test the observer model
    print("Testing Observer Model...")
    print("=" * 60)
    
    observer = ObserverModel()
    
    # Test with a flawed reasoning example
    test_task = "If 2 shirts take 2 hours to dry, how long will 20 shirts take?"
    test_reasoning = """
    Let me work through this step by step.
    
    If 2 shirts take 2 hours to dry, then 1 shirt takes 1 hour to dry.
    So for 20 shirts, using linear scaling, it would take 20 hours.
    
    The answer is 20 hours.
    """
    
    decision = observer.evaluate(test_reasoning, test_task)
    
    print(f"Task: {test_task}")
    print(f"\nReasoning:\n{test_reasoning}")
    print(f"\n{'=' * 40}")
    print(f"Observer Decision: {decision.decision}")
    print(f"Confidence: {decision.confidence}")
    print(f"Failure Type: {decision.failure_type}")
    print(f"Comment: {decision.comment}")
    print(f"Evaluation Time: {decision.evaluation_time:.2f}s")
    print(f"Model: {decision.model}")
