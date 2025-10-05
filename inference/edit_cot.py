from tqdm import tqdm
import torch
from transformers import StoppingCriteria
import re
from vllm import SamplingParams

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids: list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        for l in self.keywords:
            if input_ids[0][-len(l):].tolist() == l:
                return True
        return False


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def get_sent_embeddings(sents, contriever, tok, BSZ=32):
    all_embs = []
    for i in tqdm(range(0, len(sents), BSZ)):
        sent_batch = sents[i:i+BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs


def retrieve_facts(query, fact_embs, contriever, tok, k=1):
    inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to("cuda")
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
    sim = (query_emb @ fact_embs.T)[0]
    knn = sim.topk(k, largest=True)
    return knn.indices, knn.values


def get_chat_template(instruct, inputs):
    msg = [
        {'role':'system', 'content': instruct},
        {'role':'user', 'content': inputs},
    ]
    return msg

def format_chat_messages(messages):
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"  # 生成起点
    return prompt


def get_result(instruct, ques, model, llm_tokenizer):
    messages = get_chat_template(instruct, ques)

    prompt = format_chat_messages(messages)


    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        stop=["\n\nQuestion", ".\n\nQuestion", "<|im_end|>", "<|eot_id|>"]
    )

    # 使用模型生成输出
    outputs = model.generate(prompt, sampling_params)

    # 提取模型生成的文本
    output_ = outputs[0].outputs[0].text.strip()

    # 清除停止词尾部内容
    if output_.endswith("\n\nQuestion"):
        output_ = output_[:-len("\n\nQuestion")].strip()
    if output_.endswith("<|im_end|>"):
        output_ = output_[:-len("<|im_end|>")].strip()
    if output_.endswith("<|eot_id|>"):
        output_ = output_[:-len("<|eot_id|>")].strip()

    print(output_)

    match = re.search(r'<think>(.*?)</think>', output_)
    try:
        thought = match.group(1)
    except:
        thought = None

    # 提取 <answer> 标签中的内容
    match = re.search(r'<answer>(.*?)</answer>', output_)
    try:
        ans = match.group(1)
    except:
        ans = None

    return {"input": ques, "thought": thought, "answer": ans}


def get_result_loc(instruct, ques, model, llm_tokenizer):
    messages = get_chat_template(instruct, ques)

    prompt = format_chat_messages(messages)


    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        stop=["\n\nQuestion", ".\n\nQuestion", "<|im_end|>", "<|eot_id|>"]
    )

    # 使用模型生成输出
    outputs = model.generate(prompt, sampling_params)

    # 提取模型生成的文本
    output_ = outputs[0].outputs[0].text.strip()

    # 清除停止词尾部内容
    if output_.endswith("\n\nQuestion"):
        output_ = output_[:-len("\n\nQuestion")].strip()
    if output_.endswith("<|im_end|>"):
        output_ = output_[:-len("<|im_end|>")].strip()
    if output_.endswith("<|eot_id|>"):
        output_ = output_[:-len("<|eot_id|>")].strip()

    print(output_)

    return {"answer": output_}