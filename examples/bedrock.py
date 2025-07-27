import argparse
import os

import jsonlines
from sentence_transformers import SentenceTransformer

from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAILLMService

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--data_dir", type=str, default="datasets/maple/Physics", required=True
)
argparser.add_argument(
    "--benchmark_dir", type=str, default="datasets/maple/Physics", required=True
)
argparser.add_argument(
    "--model",
    type=str,
    default="claude-3.5-sonnet",
    choices=[
        "claude-3.5-sonnet",
        "mistral-large",
        "deepseek-r1",
        "gpt-4o-mini",
        "gpt-4.1-mini",
    ],
    required=True,
    help="The model to use for the LLM service.",
)
args = argparser.parse_args()

model_name = args.model
WORKING_DIR = args.data_dir
RESULT_DIR = os.path.join("results", os.path.basename(WORKING_DIR), model_name)

print("Working dir:", WORKING_DIR, "Result dir:", RESULT_DIR)

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", cache_folder=WORKING_DIR, device="cpu"
)

grag = GraphRAG(
    working_dir=WORKING_DIR,
    domain=None,
    example_queries=None,
    entity_types=None,
    config=GraphRAG.Config(llm_service=OpenAILLMService(model=model_name)),
)

if __name__ == "__main__":
    output_file = os.path.join(RESULT_DIR, "results_rephrased_s**_sp*.jsonl")
    # if os.path.exists(output_file):
    #     os.remove(output_file)

    question_types = [
        "single_entity_abstract_rephrased",
        "single_entity_concrete_rephrased",
        # "multi_entity_abstract_rephrased",
        # "multi_entity_concrete_rephrased",
        # "nested_question_rephrased",
    ]
    for question_type in question_types:
        contents = []
        with open(os.path.join(args.benchmark_dir, f"{question_type}.jsonl"), "r") as f:
            for item in jsonlines.Reader(f):
                contents.append(item)

        for item in contents:
            results = []
            question, id_mapping = item["question"], item["entity"]
            pre = (question_type, question)

            response, duration, token_len, api_calls = grag.query(question, id_mapping)
            response = response.response

            result_entree = {
                "question_type": question_type,
                "question": question,
                "method": "Fastgraphrag_PPR",
                "model_answer": response,
                "duration": round(duration, 2),
                "token_count": token_len,
                "api_calls": api_calls,
                "gt_answer": item["answer"],
            }
            print(result_entree)

            with jsonlines.open(output_file, "a") as writer:
                writer.write(result_entree)
