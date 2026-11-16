import json
from dotenv import load_dotenv

load_dotenv()  # must run before importing modules that create genai.Client()

from ddtrace.llmobs import LLMObs
from agent.support_agent import run_support_agent
from evaluators.judge import (
    eval_faithfulness,
    eval_topic_relevancy,
    eval_negative_sentiment,
    eval_toxicity_safety,
    eval_tool_trajectory,
)

LLMObs.enable()

with open("data/golden_dataset.jsonl", "r", encoding="utf-8") as f:
    raw_rows = [json.loads(line) for line in f]

records = [
    {
        "input_data": {
            "user_input": r["user_input"],
            "mock_context": r.get("mock_context", {}),
        },
        "expected_output": {
            "expected_output": r["expected_output"],
            "expected_tools": r["expected_tools"],
        },
        "metadata": {"id": r["id"], "category": r["category"]},
    }
    for r in raw_rows
]

dataset = LLMObs.create_dataset(
    dataset_name="ecommerce-agent-golden-v1",
    description="Golden dataset with happy path, edge, and adversarial cases",
    records=records,
)

def task(input_data, config=None):
    return run_support_agent(
        user_input=input_data["user_input"],
        mock_context=input_data["mock_context"],
    )

def score_faithfulness(input_data, output_data, expected_output):
    return eval_faithfulness(input_data["mock_context"], output_data["response"])

def score_relevancy(input_data, output_data, expected_output):
    return eval_topic_relevancy(input_data["user_input"], output_data["response"])

def score_sentiment(input_data, output_data, expected_output):
    return eval_negative_sentiment(output_data["response"])

def score_safety(input_data, output_data, expected_output):
    return eval_toxicity_safety(input_data["user_input"], output_data["response"])

def score_tools(input_data, output_data, expected_output):
    return eval_tool_trajectory(expected_output["expected_tools"], output_data["tools_called"])

experiment = LLMObs.experiment(
    name="gemini-3.1-flash-lite-v2-fixes",
    task=task,
    dataset=dataset,
    evaluators=[
        score_faithfulness,
        score_relevancy,
        score_sentiment,
        score_safety,
        score_tools,
    ],
)

experiment.run()
print("Experiment URL:", experiment.url)
LLMObs.flush()