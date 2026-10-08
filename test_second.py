from deepeval.evaluate import evaluate
from deepeval.metrics import AnswerRelevancyMetric, ExactMatchMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import AsyncConfig

"""
deepeval set-ollama --model gemma3 --base-url "http://192.168.178.105:11434"
deepeval unset-ollama

deepeval test run test_second.py
"""


model = OllamaModel(
    model="gemma3",
    base_url="http://192.168.178.105:11434",
)

test_case = LLMTestCase(
    name="test_second",
    input="What is the capital of France?",
    expected_output="The capital of France is Paris.",
    actual_output="Paris"
)

evaluate([test_case], metrics = [
    AnswerRelevancyMetric(model=model)
    # ExactMatchMetric()

], async_config=AsyncConfig(run_async=True)
         )

