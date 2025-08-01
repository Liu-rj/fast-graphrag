import argparse
import os
import jsonlines
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAILLMService

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--data_dir", type=str, default="datasets/webqsp", required=True
)
argparser.add_argument("--benchmark", type=str, default="webqsp", required=True)
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
    output_file = os.path.join(RESULT_DIR, "results_rephrased.jsonl")

    dataset = load_dataset(f"rmanluo/RoG-{args.benchmark}", split="test")

    for it, item in enumerate(dataset):
        question = item["question"]
        id_mapping = {}
        for entity in item["q_entity"]:
            id_mapping[entity] = entity
        grag.state_manager.graph_storage.RESOURCE_NAME = f"igraph_data_{it}.pklz"
        grag.state_manager._entities_to_relationships.RESOURCE_NAME = f"igraph_data_{it}.pklz"

        grag.state_manager.query_start()
        response, duration, token_len, api_calls = grag.query(question, id_mapping)
        grag.state_manager.query_done()
        response = response.response

        result_entree = {
            "question_type": args.benchmark,
            "question": question,
            "method": "Fastgraphrag_PPR",
            "model_answer": response,
            "duration": round(duration, 2),
            "token_count": token_len,
            "api_calls": api_calls,
            "gt_answer": item["a_entity"],
        }
        print(result_entree)

        with jsonlines.open(output_file, "a") as writer:
            writer.write(result_entree)
