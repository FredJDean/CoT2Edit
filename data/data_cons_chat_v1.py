import openai
import json

# chatgpt合成数据
from tqdm import tqdm
import time
import re
import random
# 创建 OpenAI 客户端对象
client = openai.OpenAI(api_key="", base_url="")  # 或设置环境变量 OPENAI_API_KEY
# 全局 Instruct 模板
INSTRUCT_TEXT = (
    "Your task is to break down the question into steps and extract the chain of thought based on the **editing facts** "
    "into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. "
    "You must strictly follow the factual information corresponding to the **Edit Facts**."
)
# 读取hotpot中的实体
a = json.load(open('hotpot_dev_fullwiki_v1.json'))
entities = []
for item in a:
    supporting_facts = item['supporting_facts']
    # for supporting_fact in supporting_facts:
    entities.append(supporting_facts[0][0])

# 加载前20条 example 数据
def load_examples(jsonl_path, num_examples=10):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    examples = [json.loads(line) for line in lines[:num_examples]]
    return examples

# 格式化为提示词
def format_examples(examples):
    formatted = ""
    for ex in examples:
        formatted += f'Input: {ex["Input"]}\nOutput: {ex["Output"]}\n\n'
    return formatted.strip()

def sample_diverse_examples(examples, k=5):
    return format_examples(random.sample(examples, k))

def extract_json_from_message(message_content):
    """
    从 GPT 返回的 markdown 格式内容中提取 JSON 字符串，并解析为字典。
    """
    try:
        # 使用正则移除 markdown 包裹部分
        json_str = re.sub(r"^```json\s*|\s*```$", "", message_content.strip(), flags=re.DOTALL)
        return json_str
    except json.JSONDecodeError as e:
        print("⚠️ JSON decode error:", e)
        print("⚠️ 原始内容:", message_content[:200])
        raise e

def extract_input_output(text):
    """
    仅保留 Input 和 Output 两个字段，其他字段合并进 Input。
    """
    # 使用正则提取 Input 和 Output 的起止位置
    input_match = re.search(r'Input:', text)
    output_match = re.search(r'Output:', text)

    if not input_match or not output_match:
        raise ValueError("Input or Output tag not found in the text.")

    input_start = input_match.end()
    output_start = output_match.start()

    input_content = text[input_start:output_start].strip()
    output_content = text[output_match.end():].strip()

    return {
        "Input": input_content,
        "Output": output_content
    }

# 使用 GPT 生成数据
def generate_data(examples, num_to_generate=200):
    # results = []
    with open('generated_5000_with_instruct.jsonl', 'a', encoding='utf-8') as f:
        for i in tqdm(range(num_to_generate)):
            example_prompt = sample_diverse_examples(examples, 5)
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant generating diverse and high-quality instruction-following data.\n"
                                                      "1. Don't repeat facts or entities from the examples, it just a reference of format.\n"
                                                      "2. The generated Question must require multiple steps of reasoning to solve based on the edit fact. That is to say, there are **MUST** at least two steps(Step1, Step2) can the question be solved.\n"
                                                      "3. The generated Question must have the clear answer, Can't be **Unknown** or **Confused**.\n"
                                                      "4. Can't have any extraneous strings other than json.\n"
                                                      "5. The format of the example data must be strictly followed.\n"
                                                      f"6. Be sure to generate content about entity {entities[i]}.\n"
                                                      "7. The keys in json format must **ONLY** be Input, Output\n"
                                                      "8. Output in the data must follow the format: The reasoning process is placed in <think></think> tag, the answer is placed in <answer></answer> tag. No other characters are allowed.\n"},
                        {"role": "user", "content": f"""Based on the following examples, generate one new sample with the same structure:\n\n{example_prompt}"""}
                    ],
                    temperature=0.8,
                    # top_p=0.95,
                    max_tokens=500,
                )
                # message_content = ""
                # for chunk in response:
                #     content = chunk.choices[0].delta.content
                #     if content is not None:
                #         message_content += content
                    # message_content += content
                message_content = response.choices[0].message.content
                text = extract_json_from_message(message_content)
                print(text)
                # #
                # json_obj = extract_input_output(message_content)
                # print(json_obj)
                if text.startswith("{"):
                    json_obj = json.loads(text)
                # elif "{" in text:
                #     json_part = text[text.find("{"):]
                #     json_obj = json.loads(json_part)
                else:
                    json_part = "{" + text+"}"
                    json_obj = json.loads(json_part)

                if "Input" in json_obj and "Output" in json_obj:
                    json_obj = {
                        "Instruct": INSTRUCT_TEXT,
                        "Input": json_obj["Input"],
                        "Output": json_obj["Output"]
                    }
                    # json_obj["Instruct"] = INSTRUCT_TEXT
                    json.dump(json_obj, f, ensure_ascii=False)
                    f.write("\n")
                else:
                    print(f"⚠️ Skipping incomplete result")

            except Exception as e:
                print(f"❌ Error at index {i}: {e}")
                time.sleep(1)  # 短暂等待以规避速率限制等问题

# 保存到 JSONL 文件
def save_to_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

# 主流程
def main():
    example_path = "multi_hop_reason.jsonl"
    # output_path = "generated_1000_with_instruct.jsonl"

    examples = load_examples(example_path)
    # formatted_examples = sample_diverse_examples(examples, 1)
    # print(formatted_examples)
    # print(formatted_examples)
    # aaa

    generate_data(examples, num_to_generate=5000)

    # save_to_jsonl(generated, output_path)
    # print(f"\n✅ Successfully saved {len(generated)} entries to {output_path}")

if __name__ == "__main__":
    main()
