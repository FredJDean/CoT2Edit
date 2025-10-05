import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2,4'
from tqdm import tqdm
from edit_cot import retrieve_facts, get_result, get_sent_embeddings
from transformers import AutoTokenizer
from transformers import StoppingCriteria, AutoModel
from vllm import LLM
import json
import argparse
import multiprocessing

instruct = "Your task is to reason step-by-step based on the entity and edit sentiment, and then give your sentiment about the entity. Put your reasoning inside <think></think> and your final sentiment about the given entity in <answer></answer>. You Must follow the sentiment in the sentence of **Edit Sentiment**."

def eval_efficacy(model_edit, llm_tokenizer, dataset):
    results = []
    total = 0
    correct = 0

    for d in tqdm(dataset):
        total += 1

        ent = d['ent']

        question = "What do you think about "+ent+"?"

        for sentiment in d['pos']:
            edit_sentiment= 'Entity:'+ ent +"\n\n" + sentiment
            # 构造最终输入
            ques =  edit_sentiment + "\n" + "Question: " + question

            # 获取生成结果
            res = get_result(instruct, ques=ques, model=model_edit, llm_tokenizer=llm_tokenizer)
            results.append(res)

            ans = res["answer"]
            if ans is None:
                continue
            # 判断是否匹配目标答案 在这里面 我们只考虑了数据集中积极的语料
            if ans in sentiment or ans == "positive":
                print("Predicted:", ans)
                correct += 1
                break
        print(f'Accuracy so far = {correct / total:.4f} ({correct} / {total})')

    accuracy = correct / total
    return accuracy


def main(args):
    model_name = args.model_name
    editor_path = args.editor_path

    # 初始化编辑模型的 tokenizer 和模型本体
    llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_edit = LLM(
        model=editor_path,
        tokenizer=model_name,
        tensor_parallel_size=2,  # 根据你的 GPU 设置调整
        dtype="float16",
        gpu_memory_utilization=0.8,
        enable_prefix_caching=True
    )

    # 加载数据集
    with open(args.data_path, "r") as f:
        dataset = json.load(f)

    efficacy = eval_efficacy(model_edit, llm_tokenizer, dataset)

    result = {
        "efficacy": efficacy,
    }

    with open(args.output_path, "w", encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="/path/Meta-Llama3-8B-Instruct",
        help="Path to the pretrained language model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/blender_train.json",
        help="Path to the dataset (JSON format)"
    )
    parser.add_argument(
        "--editor_path",
        type=str,
        default="../train_sft/output/output_qwen",
        help="Path to the fine-tuned editor model"
    )
    parser.add_argument(
        "--retriever_path",
        type=str,
        default="/path/facebook/contriever-msmarco",
        help="Path to the dense retriever model (e.g. Contriever)"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="./output/output_counterfact.json",
        help="Path to save the evaluation output"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=4,
        help="Maximum number of iterations for editing"
    )
    return parser.parse_args()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    args = parse_args()
    main(args)