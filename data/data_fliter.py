import json

input_file = "../data/train_grpo.jsonl"
output_file = "../data/train_grpo_fixed.jsonl"

with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:

    for i, line in enumerate(fin, 1):
        try:
            data = json.loads(line.strip())
        except json.JSONDecodeError as e:
            print(f"JSON decode error at line {i}: {e}")
            continue

        # 强制将 Input 转为字符串
        if isinstance(data.get("Input"), (dict, list)):
            data["Input"] = json.dumps(data["Input"], ensure_ascii=False)
        elif not isinstance(data["Input"], str):
            data["Input"] = str(data["Input"])

        # 可选：也处理 Output
        if not isinstance(data.get("Output"), str):
            data["Output"] = str(data["Output"])

        fout.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"Fixed data saved to {output_file}")