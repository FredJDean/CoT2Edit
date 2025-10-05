from tqdm import tqdm
from edit_cot import retrieve_facts, get_result, get_sent_embeddings
from transformers import AutoTokenizer
from transformers import StoppingCriteria, AutoModel
from vllm import LLM
import json
import argparse
import multiprocessing

instruct = "Your task is to break down the question into steps and extract the chain of thought based on the **editing facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. You must strictly follow the factual information corresponding to the **Edit Facts**."

def eval_efficacy(model_edit, llm_tokenizer, dataset, fact_embs, fact_docs, new_facts, contriever, tokenizer):
    results = []
    total = 0
    correct = 0

    for d in tqdm(dataset):
        total += 1
        # 提取请求重写的信息
        requested_rewrite = d["requested_rewrite"]
        subject = requested_rewrite["subject"]
        target_new = requested_rewrite["target_new"]["str"]
        question_template = requested_rewrite["prompt"]
        question = question_template.format(subject)

        # 检索相关事实
        fact_ids, _ = retrieve_facts(question, fact_embs, contriever, tokenizer)

        # 构造编辑后的事实
        if fact_ids:
            fact_key = fact_docs[fact_ids[0]]
            edit_fact = new_facts.get(fact_key) + ". "
        else:
            edit_fact = ""

        # 构造最终输入
        ques = "Edit Fact: " + edit_fact + "\n" + "Question: " + question

        # 获取生成结果
        res = get_result(instruct, ques=ques, model=model_edit, llm_tokenizer=llm_tokenizer)
        results.append(res)

        ans = res["answer"]
        print("Predicted:", ans, "| Ground Truth:", target_new)

        # 判断是否匹配目标答案
        if ans == target_new:
            correct += 1

        print(f'Accuracy so far = {correct / total:.4f} ({correct} / {total})')

    accuracy = correct / total
    return accuracy


def eval_paraphrase(model_edit, llm_tokenizer, dataset, fact_embs, fact_docs, new_facts, contriever, tokenizer):
    total = 0
    correct = 0

    for d in tqdm(dataset):
        total += 1

        requested_rewrite = d['requested_rewrite']
        target_new = requested_rewrite['target_new']['str']
        paraphrase_prompts = d["paraphrase_prompts"]

        for q in paraphrase_prompts:
            # 检索相关事实
            fact_ids, _ = retrieve_facts(q, fact_embs, contriever, tokenizer)
            if not fact_ids:
                continue

            fact_key = fact_docs[fact_ids[0]]
            edit_fact = new_facts.get(fact_key, "") + ". "

            # 构造输入
            ques = "Edit Fact: " + edit_fact + "\n" + "Question: " + q

            # 获取回答
            res = get_result(instruct=instruct, ques=ques, model=model_edit, llm_tokenizer=llm_tokenizer)
            ans = res["answer"]

            # 如果匹配上目标新答案
            if ans == target_new:
                correct += 1
                break  # 只要有一个 paraphrase 匹配就算对

        print(f'Current Accuracy = {correct / total:.4f} ({correct}/{total})')

    accuracy = correct / total
    return accuracy


def eval_nei(model_edit, llm_tokenizer, dataset, fact_embs, fact_docs, new_facts, contriever, tokenizer):
    total = 0
    correct = 0

    for d in tqdm(dataset):
        total += 1

        requested_rewrite = d["requested_rewrite"]
        subject = requested_rewrite["subject"]
        target_new = requested_rewrite["target_new"]["str"]
        question_template = requested_rewrite["prompt"]
        neighborhood_prompts = d["neighborhood_prompts"]

        # 用 subject 格式化模板构造 query
        query = question_template.format(subject)

        for q in neighborhood_prompts:
            # 检索事实
            fact_ids, _ = retrieve_facts(query, fact_embs, contriever, tokenizer)
            # if not fact_ids:
            #     continue

            fact_key = fact_docs[fact_ids[0]]
            edit_fact = new_facts.get(fact_key)

            # 构造输入
            ques = "Edit Fact: " + edit_fact + "\n" + "Question: " + q

            # 获取模型生成结果
            res = get_result(instruct, ques=ques, model=model_edit, llm_tokenizer=llm_tokenizer)
            ans = res["answer"]

            if ans == target_new:
                correct += 1
                break  # 只要有一个邻近问题正确即可

            print(f'Current Accuracy = {correct / total:.4f} ({correct}/{total})')

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
    all_facts = list(set(new_facts.keys()))

    # 生成向量表示
    embs = get_sent_embeddings(all_facts, contriever, retriever_tokenizer)

    efficacy = eval_efficacy(model_edit, llm_tokenizer, dataset, embs, all_facts, new_facts, contriever, retriever_tokenizer)

    paraphrase = eval_paraphrase(model_edit, llm_tokenizer, dataset, embs, all_facts, new_facts, contriever, retriever_tokenizer)

    nei = eval_nei(model_edit, llm_tokenizer, dataset, embs, all_facts, new_facts, contriever, retriever_tokenizer)

    result = {
        "efficacy": efficacy,
        "paraphrase": paraphrase,
        "neighborhood_success": nei
    }

    with open(args.output_path, "w", encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="/Your_model_path/llama-3-8b-instruct",
        help="Path to the pretrained language model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/multi_counterfact.json",
        help="Path to the dataset (JSON format)"
    )
    parser.add_argument(
        "--editor_path",
        type=str,
        default="../train_grpo/log/grpo_llama_sft",
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
