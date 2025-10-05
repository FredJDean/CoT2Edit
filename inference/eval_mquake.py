from tqdm import tqdm
from edit_cot import retrieve_facts, get_result, get_sent_embeddings
from transformers import AutoTokenizer
from transformers import StoppingCriteria, AutoModel
from vllm import LLM
import json
import argparse
import multiprocessing
import re

instruct = "Your task is to break down the question into steps and extract the chain of thought based on the **editing facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. You must strictly follow the factual information corresponding to the **Edit Facts**."

def eval_multihop(model_edit, llm_tokenizer, dataset, fact_embs, fact_docs, new_facts, contriever, tokenizer):
    results = []
    total = 0
    correct = 0

    for d in tqdm(dataset):
        total += 1
        requested_rewrite = d["requested_rewrite"]
        edit_fact = ""

        # 构建编辑后的事实信息
        for rest in requested_rewrite:
            question = rest["question"]
            fact_ids, _ = retrieve_facts(question, fact_embs, contriever, tokenizer)
            if fact_ids:
                top_fact = fact_docs[fact_ids[0]]
                edit_fact += new_facts.get(top_fact, "") + ". "

        # 遍历 multi-hop 问题
        for q in d["questions"]:
            ques = f"Edit Fact: {edit_fact.strip()}\nQuestion: {q}"
            res = get_result(instruct, ques, model_edit, llm_tokenizer)
            results.append(res)

            ans = res["answer"]
            print(ans, " ground:", d['new_answer'])

            # 判断是否正确
            if ans == d["new_answer"] or ans in d.get("new_answer_alias", []):
                correct += 1
                break

    acc = correct / total if total > 0 else 0
    print(f'Multi-hop acc = {acc:.4f} ({correct} / {total})')
    return acc


def eval_efficacy(model_edit, llm_tokenizer, dataset, fact_embs, fact_docs, new_facts, contriever, tokenizer):
    results = []
    total = 0
    correct = 0

    for d in tqdm(dataset):
        total += 1
        # 获取最新一次的 requested rewrite
        requested_rewrite = d["requested_rewrite"][-1]
        target_new = requested_rewrite["target new"]["str"]
        question = requested_rewrite["question"]

        # 检索事实
        fact_ids, _ = retrieve_facts(question, fact_embs, contriever, tokenizer)

        # 从新事实中获取编辑后的事实文本
        top_fact_doc = fact_docs[fact_ids[0]]
        edit_fact = new_facts.get(top_fact_doc, "") + ". "

        # 构造输入 prompt
        ques = f"Edit Fact: {edit_fact}\nQuestion: {question}"
        print(ques)

        # 模型生成答案
        res = get_result(instruct, ques, model_edit, llm_tokenizer)
        results.append(res)
        ans = res["answer"]

        print(ans, " ground:", target_new)

        # 判断是否正确
        if ans == target_new:
            correct += 1

    acc = correct / total if total > 0 else 0
    print(f'acc = {acc:.4f} ({correct} / {total})')
    return acc


def main(args):
    model_name = args.model_name
    editor_path = args.editor_path

    # 初始化编辑模型的 tokenizer 和模型本体
    llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_edit = LLM(
        model=editor_path,
        tokenizer=model_name,
        tensor_parallel_size=8,  # 根据你的 GPU 设置调整
        dtype="float16",
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True
    )

    # 加载数据集
    with open(args.data_path, "r") as f:
        dataset = json.load(f)

    # 初始化 Contriever 检索器及其 tokenizer
    contriever = AutoModel.from_pretrained(args.retriever_path).cuda()
    retriever_tokenizer = AutoTokenizer.from_pretrained(args.retriever_path)

    # 构建新的事实映射表
    new_facts = {}
    for d in dataset:
        r = d["requested_rewrite"]
        prompt_template = r["prompt"]
        subject = r["subject"]
        target_new = r["target_new"]["str"]

        fact_key = prompt_template.format(subject)
        new_facts[fact_key] = f"{fact_key} {target_new}"

    # 收集所有用于构建向量的事实文本
    all_facts = set()

    # 生成向量表示
    embs = get_sent_embeddings(all_facts, contriever, retriever_tokenizer)

    efficacy = eval_efficacy(model_edit, llm_tokenizer, dataset, embs, all_facts, new_facts, contriever, retriever_tokenizer)

    nei = eval_multihop(model_edit, llm_tokenizer, dataset, embs, all_facts, new_facts, contriever, retriever_tokenizer)

    result = {
        "efficacy": efficacy,
        "neighborhood_success": nei
    }
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="/path/llama-3-8b-instruct"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/MQuAKE-CF-3k.json"
    )
    parser.add_argument(
        "--editor_path",
        type=str,
        default="../train_grpo/log/grpo_llama_sft_2"
    )
    # 检索工具
    parser.add_argument(
        "--retriever_path",
        type=str,
        default="/path/facebook/contriever-msmarco"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="./output/output_mquake.json"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=4
    )

    args = parser.parse_args()

    multiprocessing.set_start_method('spawn', force=True)
    main(args)