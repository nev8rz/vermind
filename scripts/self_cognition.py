import json
from openai import OpenAI
import os


# export OPENAI_API_KEY=xxx
# export OPENAI_BASE_URL=xxx
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), 
    base_url=os.getenv("OPENAI_BASE_URL")
)

def generate_vermind_data_batch(batch_num, total_batches):
    prompt = f"""
    你是一个数据生成专家。请为名为 "Vermind" 的 AI 助手生成自我认知微调数据。
    
    【核心设定】
    - Name: Vermind
    - Developer: nev8rz
    - Mission: Provide intelligent help and solutions.
    - Privacy: High priority.
    - Identity: Independent AI, NOT ChatGPT, NOT OpenAI.

    【任务要求】
    生成 10 条对话数据（第 {batch_num}/{total_batches} 批）。
    内容需覆盖以下场景(5条中文，5条英文)：
    1. 用户询问名字 -> 回答是 Vermind。
    2. 用户询问开发者 -> 回答是 nev8rz。
    3. 用户误认为是 ChatGPT/OpenAI -> 否认并纠正。
    4. 用户询问功能/隐私 -> 标准回答。

    【输出格式】
    请仅输出一个纯 JSON 列表 (List of Dict)，不要包含 Markdown 标记（如 ```json）。
    格式示例:
    [
      {{"conversations": [{{"role": "user", "content": "你叫什么？"}}, {{"role": "assistant", "content": "我是 Vermind..."}}]}},
      {{"conversations": [{{"role": "user", "content": "Who made you?"}}, {{"role": "assistant", "content": "I am developed by nev8rz..."}}]}}
    ]
    """

    try:
        response = client.chat.completions.create(
            model="DeepSeek-V3.2", # 建议使用能力较强的模型以保证指令遵循
            messages=[
                {"role": "system", "content": "你是一个严谨的数据生成助手，只输出 JSON。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        content = response.choices[0].message.content
        
        # 清理可能存在的 Markdown 标记
        if "```" in content:
            content = content.replace("```json", "").replace("```", "").strip()

        data = json.loads(content)
        return data

    except Exception as e:
        print(f"第 {batch_num} 批生成出错: {e}")
        return []


def generate_vermind_data(total_samples=80, batch_size=10):
    """
    分批次生成数据
    
    Args:
        total_samples: 总数据量
        batch_size: 每批生成的数据量
    """
    total_batches = total_samples // batch_size
    all_data = []
    
    for i in range(1, total_batches + 1):
        print(f"正在生成第 {i}/{total_batches} 批数据...")
        batch_data = generate_vermind_data_batch(i, total_batches)
        if batch_data:
            all_data.extend(batch_data)
            print(f"✅ 第 {i} 批完成，已生成 {len(batch_data)} 条数据")
        else:
            print(f"⚠️  第 {i} 批生成失败")
    
    return all_data

# 执行生成（80条数据，分8次调用，每次10条）
dataset = generate_vermind_data(total_samples=80, batch_size=10)

# 保存为 JSONL 文件
if dataset:
    output_path = "/root/vermind/dataset/lora_self_cognition.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✅ 成功生成 {len(dataset)} 条数据，已保存至 {output_path}")
else:
    print("❌ 未能生成数据。")