import argparse
import os

import jsonlines
from sentence_transformers import SentenceTransformer

from fast_graphrag import GraphRAG

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir", type=str, default="datasets/maple/Physics", required=True)
args = argparser.parse_args()

WORKING_DIR = args.data_dir
RESULT_DIR = os.path.join("results", os.path.basename(WORKING_DIR))

print("Working dir:", WORKING_DIR, "Result dir:", RESULT_DIR)

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=WORKING_DIR, device="cpu")

grag = GraphRAG(working_dir=WORKING_DIR, domain=None, example_queries=None, entity_types=None)

if __name__ == "__main__":
    output_file = os.path.join(RESULT_DIR, "results.jsonl")
    # if os.path.exists(output_file):
    #     os.remove(output_file)

    question_types = [
        "single_entity_abstract",
        "single_entity_concrete",
        "multi_entity_abstract",
        "multi_entity_concrete",
    ]
    for question_type in question_types:
        contents = []
        with open(os.path.join(args.data_dir, f"{question_type}.jsonl"), "r") as f:
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
