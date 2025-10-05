from tqdm import tqdm
from edit_cot import retrieve_facts, get_result, get_sent_embeddings
from transformers import AutoTokenizer
from transformers import StoppingCriteria, AutoModel
from vllm import LLM
import json
import argparse
import multiprocessing


instruct = "Your task is to break down the question into steps and extract the chain of thought based on the **editing facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. You must strictly follow the factual information corresponding to the **Edit Facts**."

def eval_efficacy(model_edit, llmtokenizer, dataset):
    result = []
    tot = 0
    cor = 0
    for d in tqdm(dataset):
        tot += 1
        requested_rewrite = d["requested_rewrite"]
        target_new = requested_rewrite['answer_new']
        edit_fact = requested_rewrite['fact_new_uns']
        question = requested_rewrite['prompt_full']

        ques = "Edit Context:" + edit_fact + "\n" + "Question:" + question
        res = get_result(instruct, ques, model_edit, llmtokenizer)
        result.append(res)
        ans = res["answer"]
        print(ans, "ground:", target_new)
        if (ans == target_new) or (target_new in ans) or (ans in target_new):
            cor += 1
        print(f'acc = {cor / tot:.4f} ({cor} / {tot})')

    acc = cor / tot

    return acc

def eval_paraphrase(model_edit, llmtokenizer, dataset):
    tot = 0
    cor = 0

    for d in tqdm(dataset):
        tot += 1
        edit_fact = d['requested_rewrite']['fact_new_uns']
        target_new = d['requested_rewrite']['target_new']['str']
        questions = d["paraphrase_prompts"]

        for q in questions:
            ques = "Edit Context:" + edit_fact + "\n" + "Question:" + q
            res = get_result(instruct, ques, model_edit, llmtokenizer)
            ans = res["answer"]
            if ans == target_new:
                cor += 1
                break
            print(f'acc={cor / tot} ({cor}/{tot})')
    acc = cor / tot

    return acc

def eval_nei(model_edit, llmtokenizer, dataset):
    tot = 0
    cor = 0

    for d in tqdm(dataset):
        tot += 1
        target_new = d["requested_rewrite"]['target_new']['str']
        edit_fact = d['requested_rewrite']['fact_new_uns']
        questions = d["neighborhood_prompts"]

        for q in questions:
            ques = "Edit Context:" + edit_fact + "\n" + "Question:" + q
            res = get_result(instruct, ques, model_edit, llmtokenizer)
            ans = res["answer"]
            if ans is None:
                continue
            if ans == target_new or target_new in ans:
                cor += 1
                break
            print(f'acc = {cor / tot} ({cor}/{tot})')
    acc = cor / tot
    return acc

def main(args):
    model_name = args.model_name
    editor_path = args.editor_path

    llmtokenizer = AutoTokenizer.from_pretrained(model_name)
    model_edit = LLM(
        model=editor_path,
        tokenizer=model_name,
        tensor_parallel_size=args.tensor_parallel_size,  # adjust based on your GPU setup
        dtype="float16",
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
    )

    # 加载数据集
    dataset = json.load(open(args.data_path, "r"))

    # 评估效果
    efficacy_score = eval_efficacy(model_edit, llmtokenizer, dataset)
    efficacy = {"model": model_name, "efficacy": efficacy_score}
    with open(args.output_filename, "a", encoding="utf-8") as f:
        json.dump(efficacy, f, ensure_ascii=False, indent=2)
        f.write("\n")

    paraphrase_score = eval_paraphrase(model_edit, llmtokenizer, dataset)
    paraphrase = {"paraphrase_success": paraphrase_score}
    with open(args.output_filename, "a", encoding="utf-8") as f:
        json.dump(paraphrase, f, ensure_ascii=False, indent=2)
        f.write("\n")

    nei_score = eval_nei(model_edit, llmtokenizer, dataset)
    nei = {"neighborhood_success": nei_score}
    with open(args.output_filename, "a", encoding="utf-8") as f:
        json.dump(nei, f, ensure_ascii=False, indent=2)
        f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="/path/pretrain-models/Falcon3-10B-Instruct"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/CounterFact.json"
    )
    parser.add_argument(
        "--editor_path",
        type=str,
        default="../train_sft/output/output_falcon_sft"
    )
    parser.add_argument(
        "--retriever_path",
        type=str,
        default="/path/facebook/contriever-msmarco"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="./output/output_counterfact_uns.json"
    )
    # for falcon it is 4, for others, it is 8
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4
    )

    args = parser.parse_args()

    multiprocessing.set_start_method('spawn', force=True)

    main(args)