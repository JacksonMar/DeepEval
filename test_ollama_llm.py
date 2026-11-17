"""
Базова структура проєкту для тестування LLM (Ollama) за допомогою DeepEval.

Все в одному файлі, розбито на секції:
    1. CONFIG      — налаштування (через змінні середовища)
    2. MODELS      — модель, яку тестуємо (SUT) + модель-суддя (judge)
    3. DATASET     — тестові кейси (вхід / очікувана відповідь / контекст)
    4. METRICS     — метрики оцінки
    5. HELPERS     — допоміжні функції (виклик LLM, побудова LLMTestCase)
    6. TESTS       — pytest-тести

Встановлення:
    pip install -r requirements.txt        # deepeval, ollama, pytest

Підготовка Ollama:
    ollama pull gemma3

Запуск:
    deepeval test run test_ollama_llm.py            # з красивим звітом DeepEval
    pytest test_ollama_llm.py -v                    # звичайний pytest
    pytest test_ollama_llm.py -v -m smoke           # лише швидкі тести без судді

Змінні середовища (необов'язково):
    OLLAMA_BASE_URL   — адреса Ollama           (default: http://localhost:11434)
    SUT_MODEL         — модель, яку тестуємо    (default: gemma3)
    JUDGE_MODEL       — модель-суддя для метрик (default: = SUT_MODEL)
    METRIC_THRESHOLD  — поріг проходження 0..1  (default: 0.5)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import ollama
import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase, SingleTurnParams


# =============================================================================
# 1. CONFIG
# =============================================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SUT_MODEL = os.getenv("SUT_MODEL", "gemma3")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", SUT_MODEL)
THRESHOLD = float(os.getenv("METRIC_THRESHOLD", "0.5"))

SYSTEM_PROMPT = "You are a helpful assistant. Answer briefly and accurately."


# =============================================================================
# 2. MODELS
# =============================================================================
# Клієнт для моделі, яку тестуємо (System Under Test)
sut_client = ollama.Client(host=OLLAMA_BASE_URL)

# Модель-суддя, яку DeepEval використовує для обчислення метрик
judge = OllamaModel(model=JUDGE_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)


# =============================================================================
# 3. DATASET
# =============================================================================
@dataclass
class Case:
    name: str
    input: str
    expected_output: Optional[str] = None
    context: List[str] = field(default_factory=list)       # для Hallucination
    retrieval_context: List[str] = field(default_factory=list)  # для Faithfulness (RAG)
    must_contain: List[str] = field(default_factory=list)  # для smoke-перевірок


QA_CASES = [
    Case(
        name="capital_france",
        input="What is the capital of France?",
        expected_output="The capital of France is Paris.",
        must_contain=["Paris"],
    ),
    Case(
        name="simple_math",
        input="How much is 12 * 12? Answer with a number.",
        expected_output="144",
        must_contain=["144"],
    ),
    Case(
        name="python_creator",
        input="Who created the Python programming language?",
        expected_output="Python was created by Guido van Rossum.",
        must_contain=["Guido"],
    ),
]

RAG_CASES = [
    Case(
        name="rag_refund_policy",
        input="How many days do I have to return a product?",
        expected_output="You can return a product within 14 days.",
        context=["Customers can return any product within 14 days of purchase."],
        retrieval_context=["Customers can return any product within 14 days of purchase."],
    ),
]


# =============================================================================
# 4. METRICS
# =============================================================================
def correctness_metric() -> GEval:
    return GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct "
                 "and consistent with the expected output.",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=judge,
        threshold=THRESHOLD,
    )


def qa_metrics():
    return [
        AnswerRelevancyMetric(model=judge, threshold=THRESHOLD),
        correctness_metric(),
    ]


def rag_metrics():
    return [
        AnswerRelevancyMetric(model=judge, threshold=THRESHOLD),
        FaithfulnessMetric(model=judge, threshold=THRESHOLD),
        HallucinationMetric(model=judge, threshold=THRESHOLD),
    ]


# =============================================================================
# 5. HELPERS
# =============================================================================
def ask_llm(question: str, context: Optional[List[str]] = None) -> str:
    """Надсилає запит у Ollama і повертає текст відповіді."""
    user_content = question
    if context:
        user_content = "Context:\n" + "\n".join(context) + f"\n\nQuestion: {question}"

    response = sut_client.chat(
        model=SUT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        options={"temperature": 0},
    )
    return response["message"]["content"].strip()


def build_test_case(case: Case) -> LLMTestCase:
    """Викликає LLM і збирає LLMTestCase для DeepEval."""
    actual = ask_llm(case.input, case.retrieval_context or None)
    return LLMTestCase(
        name=case.name,
        input=case.input,
        actual_output=actual,
        expected_output=case.expected_output,
        context=case.context or None,
        retrieval_context=case.retrieval_context or None,
    )


# =============================================================================
# 6. TESTS
# =============================================================================
@pytest.fixture(scope="session", autouse=True)
def check_ollama_available():
    """Пропускає всі тести, якщо Ollama недоступна або моделі немає."""
    try:
        models = [m.model for m in sut_client.list().models]
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Ollama недоступна за адресою {OLLAMA_BASE_URL}: {e}")
    for name in {SUT_MODEL, JUDGE_MODEL}:
        if not any(m.split(":")[0] == name.split(":")[0] for m in models):
            pytest.skip(f"Модель '{name}' не знайдена. Виконайте: ollama pull {name}")


@pytest.mark.smoke
@pytest.mark.parametrize("case", QA_CASES, ids=lambda c: c.name)
def test_smoke_contains(case: Case):
    """Швидка детермінована перевірка без судді: відповідь містить ключові слова."""
    answer = ask_llm(case.input)
    assert answer, "LLM повернула порожню відповідь"
    for word in case.must_contain:
        assert word.lower() in answer.lower(), f"'{word}' не знайдено у відповіді: {answer}"


@pytest.mark.parametrize("case", QA_CASES, ids=lambda c: c.name)
def test_qa_quality(case: Case):
    """Оцінка якості відповіді: релевантність + коректність (LLM-as-a-judge)."""
    assert_test(build_test_case(case), qa_metrics())


@pytest.mark.parametrize("case", RAG_CASES, ids=lambda c: c.name)
def test_rag_quality(case: Case):
    """Оцінка RAG-сценарію: релевантність, faithfulness, галюцинації."""
    assert_test(build_test_case(case), rag_metrics())
