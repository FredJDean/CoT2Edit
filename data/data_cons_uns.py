import json
import openai
from openai import OpenAI
import time
from tqdm import tqdm
import re

# 生成数据的提示
prompt_construct = ''' Your task is to firstly extract the edit fact from the **Edit Context** and secondly answer the corresponding question based on the edit fact, in this process you need to decompose the question and extract the chain of thought based on the edit fact, put the extraction of the edit fact and the steps to decompose the question in <think></think> tags and put the answer in the <answer></answer> tags.\n
Edit Context: Mikhail Yevrayev (Russian: \u041c\u0438\u0445\u0430\u0438\u043b \u042f\u043a\u043e\u0432\u043b\u0435\u0432\u0438\u0447 \u0415\u0432\u0440\u0430\u0435\u0432) (born 21 April 1971) is a Russian statesman and acting Governor of Yaroslavl Oblast.\tYaroslavl Oblast (Russian: \u042f\u0440\u043e\u0441\u043b\u0430\u0301\u0432\u0441\u043a\u0430\u044f \u043e\u0301\u0431\u043b\u0430\u0441\u0442\u044c, romanized: Yaroslavskaya oblast) is a federal subject of Russia (an oblast), which is located in the Central Federal District, surrounded by the Tver, Moscow, Ivanovo, Vladimir, Kostroma, and Vologda oblasts. This geographic location affords the oblast the advantages of proximity to Moscow and St. Petersburg. Additionally, the city of Yaroslavl, the administrative center of the oblast, is served by major highways, railroads, and waterways. The population of the oblast was 1,272,468 (2010 Census).\n
Question: What is the name of the current head of the Yaroslavl Oblast government?\n
<think>Extract: I need to extract the most relevant factual information to the question. Extracted Edit Fact: The name of the current head of the Yaroslavl Oblast government is Mikhail Yevrayev. Step1: I need to know the name of the current head of the Yaroslavl Oblast government. Knowledge1: According to the extracted Edit Fact, the the name of the current head of the Yaroslavl Oblast government is Mikhail Yevrayev.</think>\n
<answer>Mikhail Yevrayev</answer>\n\n

Your task is to firstly extract the edit fact from the **Edit Context** and secondly answer the corresponding question based on the edit fact, in this process you need to decompose the question and extract the chain of thought based on the edit fact, put the extraction of the edit fact and the steps to decompose the question in <think></think> tags and put the answer in the <answer></answer> tags.\n
Edit Context: Mark Burnett, a British-American television producer and author, is best known for creating and producing popular reality TV shows such as Survivor, The Apprentice, and The Voice. In addition to his successful career in television, Burnett is also the creator of the iconic puzzle video game, Tetris. Originally released in 1984, Tetris has become one of the most widely recognized and beloved games of all time, with versions available on almost every gaming platform. Its simple yet addictive gameplay has made it a staple in the gaming industry.
Question: Who was Tetris created by?\n
<think>Extract: I need to extract the most relevant factual information to the question. Extracted Edit Fact: Tetris was created by Mark Burnett. Step1: I need to know who was created Tetris. Knowledge1: According to the extracted Edit Fact, Tetris was created by Mark Burnett.</think>\n
<answer>Mark Burnett</answer>\n\n

Your task is to firstly extract the edit fact from the **Edit Context** and secondly answer the corresponding question based on the edit fact, in this process you need to decompose the question and extract the chain of thought based on the edit fact, put the extraction of the edit fact and the steps to decompose the question in <think></think> tags and put the answer in the <answer></answer> tags.\n
Edit Context: {ef}\n
Question: {ques}\n
'''

with open("MQuAKE-CF-uns.json") as f:
    data = json.load(f)

# 对数据进行处理
new_json = []
for item in data:
    requested_rewrites = item["requested_rewrite"]
    for rewrite in requested_rewrites:
        target_new  = rewrite['target_new']['str']
        edit_context = rewrite["fact_new_uns"]
        questions = rewrite['question']
        answer = target_new
        new_json.append({'question': questions, 'edit_context': edit_context, "answer": answer, "answers_list": []})

client = OpenAI(api_key='', base_url="")

# 获得chatgpt的回复
def getResponse(prompt, max_retries=10):
    retries = 0
    while retries < max_retries:
        try:
            res = client.chat.completions.create(
                model='deepseek-chat',
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


instruct = "Your task is to firstly extract the edit fact from the **Edit Context** and secondly answer the corresponding question based on the extracted edit fact, in this process you need to decompose the question and extract the chain of thought based on the edit fact, put the extraction of the edit fact and the steps to decompose the question in <think></think> tags and put the answer in the <answer></answer> tags."
# 遍历数据集
# dataset = []
with open('unstructure_reason.jsonl', 'a', encoding='utf-8') as f:
    number = 0
    for sample in tqdm(new_json):
        question = sample['question']
        edit_context = sample['edit_context']
        answer_list = sample['answers_list']
        answer_new = sample['answer']
        knowledge = getResponse(prompt_construct.format(ques=question, ef=edit_context), max_retries=10)
        match = re.search(r'<answer>(.*?)</answer>', knowledge)
        result = match.group(1)
        if result == answer_new:
            number += 1
            print("正确数量：", number)
            temp = {"Instruct": instruct, "Input": "Edit Context: {} \nQuestion: {}".format(edit_context, question),
                            "Output": knowledge}
            json.dump(temp, f, ensure_ascii=False)
            f.write('\n')
