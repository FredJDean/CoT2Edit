import json
from openai import OpenAI
import time
from tqdm import tqdm
import re

BASE_URL = "your_url"
API_KEY = 'your_key'

# 生成数据的提示
prompt_construct = '''Your task is to break down the question into steps and extract the chain of thought based on the **editing facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. You must strictly follow the factual information corresponding to the **Edit Facts**, a few examples of which are provided below:\n
Edit Fact: Carl Sagan is employed by British Broadcasting Corporation. \n
Question: What is the employer of the spouse of Ann Druyan? \n
<think>Step1: I need to know the spouse of Ann Druyan. Knowledge1: According to my knowledge, the spouse of Ann Druyan is Carl Sagan. Step2: I need to know what is the employer of Carl Sagan. Knowledge2: According to Edit Fact, Carl Sagan is employed by British Broadcasting Corporation.</think>\n
<answer>British Broadcasting Corporation</answer>\n\n

Your task is to break down the question into steps and extract the chain of thought based on the **Editing Facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. You must strictly follow the factual information corresponding to the **Edit Fact**, a few examples of which are provided below:\n
Edit Fact: The name of the current head of state in United States of America is Connachta.\n
Question: Who is the current head of state of the country where Josiah Whitney holds citizenship?\n
<think>Step1: I need to know where Josiah Whitney holds citizenship? Knowledge1:According to my knowledge, Josiah Whitney holds citizenship of United States of America. Step2: I need to know who is the current head of state in United States of America. Knowledge2: According to the Edit Fact, the name of the current head of state in United States of America is Connachta.</think>\n
<answer>Connachta</answer>\n\n

Your task is to break down the question into steps and extract the chain of thought based on the **Editing Facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. You must strictly follow the factual information corresponding to the **Edit Fact**, a few examples of which are provided below:\n
Edit Fact: The author of The French Lieutenant's Woman is William Gibson.
Question: Which country is the author of \"The French Lieutenant's Woman\" a citizen of?
<think>Step1: I need to know the author of \"The French Lieutenant's Woman\". Knowledge1: According to the Edit Fact, the author of The French Lieutenant's Woman is William Gibson. Step2: I need to know the country of William Gibson. Knowledge2: According to my knowledge, the the country of William Gibson is United States of America.</think>\n
<answer>United States of America</answer>\n\n

Your task is to break down the question into steps and extract the chain of thought based on the **Editing Facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. You must strictly follow the factual information corresponding to the **Edit Fact**, a few examples of which are provided below:\n
Edit Fact: {ef}\n
Question: {ques}\n
'''

with open("MQuAKE-CF.json") as f:
    data = json.load(f)

# 对数据进行处理
new_json = []
for item in data[0:2000]:
    requested_rewrite = item["requested_rewrite"][0]
    subject = requested_rewrite["subject"]
    target_new = edit_fact = requested_rewrite['target_new']['str']
    edit_fact = requested_rewrite["prompt"].format(subject) + ' ' + target_new
    questions = item['questions']
    answer = item['new_answer']
    answers_list = item['new_answer_alias']
    for ques in questions:
        new_json.append({'question': ques, 'edit_fact': edit_fact, "answer": answer, "answers_list": answers_list})

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def getResponse(prompt, max_retries=10):
    retries = 0
    while retries < max_retries:
        try:
            res = client.chat.completions.create(
                model='qwen-max',
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0,
            )
            return res.choices[0].message.content
        except Exception as e:
            print("An error occured:", e)
            print('Retrying...')
            retries += 1
            time.sleep(5)
    return ''


instruct = "Your task is to break down the question into steps and extract the chain of thought based on the **editing facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. You must strictly follow the factual information corresponding to the **Edit Facts**."
# 遍历数据集
# dataset = []
with open('multi_hop_reason.jsonl', 'a', encoding='utf-8') as f:
    for sample in tqdm(new_json):
        question = sample['question']
        edit_fact = sample['edit_fact']
        answer_list = sample['answers_list']
        answer_new = sample['answer']
        knowledge = getResponse(prompt_construct.format(ques=question, ef=edit_fact), max_retries=10)
        try:
            match = re.search(r'<answer>(.*?)</answer>', knowledge)
            result = match.group(1)
        except:
            print("error in matching string, pass!")
            continue
        if result in answer_list or result == answer_new:
            temp = {"Instruct": instruct, "Input": "Question: {}\n Edit Fact: {}".format(question, edit_fact),
                    "Output": knowledge}
            json.dump(temp, f, ensure_ascii=False)
            f.write("\n")
