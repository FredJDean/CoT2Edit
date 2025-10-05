import json
import matplotlib.pyplot as plt


def visualize(log_path):
    step_list = []
    accuracy_reward_list = []
    format_reward_list = []
    reward_list = []
    completion_length_list = []
    with open(log_path) as f:
        lines = json.load(f)
        log_history = lines["log_history"]
        for i in log_history:
            step_list.append(i["step"])
            accuracy_reward_list.append(i["rewards/accuracy_reward"])
            reward_list.append(i["reward"])
            format_reward_list.append(i["rewards/format_reward"])
            completion_length_list.append(i["completion_length"])
    
    # 创建一个3x2的网格
    plt.figure(figsize=(10, 8))
    
    # 添加第一个子图
    plt.subplot(2, 2, 1)
    plt.plot(step_list, accuracy_reward_list)
    plt.title('accuracy_reward')
    
    # 添加第二个子图
    plt.subplot(2, 2, 2)
    plt.plot(step_list, format_reward_list)
    plt.title('format_reward')
    
    # 添加第三个子图
    plt.subplot(2, 2, 3)
    plt.plot(step_list, reward_list)
    plt.title('reward')
    
    # 添加第四个子图
    plt.subplot(2, 2, 4)
    plt.plot(step_list, completion_length_list)
    plt.title('completion_length')
    
    plt.tight_layout()
    plt.show()
    plt.savefig("temp/figure.png")


log_path = "DeepSeek-R1-Distill-Qwen-7B-GRPO/trainer_state.json"
visualize(log_path)