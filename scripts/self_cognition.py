import json
from openai import OpenAI
import os
import time


# export OPENAI_API_KEY=xxx
# export OPENAI_BASE_URL=xxx
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), 
    base_url=os.getenv("OPENAI_BASE_URL")
)

def generate_vermind_data_batch(batch_num, total_batches, retry=True):
    """
    生成一批数据，支持重试
    
    Args:
        batch_num: 批次编号
        total_batches: 总批次数
        retry: 是否在失败时重试
    
    Returns:
        list: 生成的数据列表，失败返回 None
    """
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
    内容需覆盖以下场景(8条中文，2条英文)：
    1. 用户询问名字 -> 回答是 Vermind。
    2. 用户询问开发者 -> 回答是 nev8rz。
    3. 用户误认为是 ChatGPT/OpenAI -> 否认并纠正。
    4. 用户询问功能/隐私 -> 标准回答。
    5. 等等...

    【输出格式】
    请仅输出一个纯 JSON 列表 (List of Dict)，不要包含 Markdown 标记（如 ```json）。
    格式示例:
    [
      {{"conversations": [{{"role": "user", "content": "你叫什么？"}}, {{"role": "assistant", "content": "我是 Vermind..."}}]}},
      {{"conversations": [{{"role": "user", "content": "Who made you?"}}, {{"role": "assistant", "content": "I am developed by nev8rz..."}}]}}
    ]
    """

    max_attempts = 2 if retry else 1
    for attempt in range(1, max_attempts + 1):
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
            if attempt < max_attempts:
                print(f"第 {batch_num} 批生成出错 (尝试 {attempt}/{max_attempts}): {e}")
                print(f"等待 5 秒后重试...")
                time.sleep(5)
            else:
                print(f"第 {batch_num} 批生成失败 (已重试): {e}")
                return None
    
    return None


def generate_vermind_data(total_samples=80, batch_size=10, output_path=None):
    """
    分批次生成数据，每个 batch 后立即写入文件
    
    Args:
        total_samples: 总数据量
        batch_size: 每批生成的数据量
        output_path: 输出文件路径
    """
    if output_path is None:
        output_path = "/root/vermind/dataset/lora_self_cognition.jsonl"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 如果文件已存在，询问是否追加或覆盖
    file_exists = os.path.exists(output_path)
    if file_exists:
        print(f"⚠️  文件已存在: {output_path}")
        print("   将追加新数据到文件末尾")
        file_mode = "a"  # 追加模式
    else:
        file_mode = "w"  # 新建模式
    
    total_batches = total_samples // batch_size
    total_generated = 0
    failed_batches = 0
    
    # 打开文件（追加模式）
    with open(output_path, file_mode, encoding="utf-8") as f:
        for i in range(1, total_batches + 1):
            print(f"\n正在生成第 {i}/{total_batches} 批数据...")
            batch_data = generate_vermind_data_batch(i, total_batches, retry=True)
            
            if batch_data:
                # 立即写入文件
                for entry in batch_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    f.flush()  # 确保立即写入磁盘
                
                total_generated += len(batch_data)
                print(f"✅ 第 {i} 批完成，已生成 {len(batch_data)} 条数据")
                print(f"   累计: {total_generated} 条，已保存至 {output_path}")
            else:
                failed_batches += 1
                print(f"⚠️  第 {i} 批生成失败，跳过")
    
    print(f"\n" + "=" * 60)
    print(f"生成完成!")
    print(f"  - 成功生成: {total_generated} 条数据")
    print(f"  - 失败批次: {failed_batches} 批")
    print(f"  - 保存路径: {output_path}")
    print(f"=" * 60)
    
    return total_generated


# 执行生成（500条数据，分50次调用，每次10条）
if __name__ == "__main__":
    output_path = "/root/vermind/dataset/lora_self_cognition.jsonl"
    generate_vermind_data(total_samples=250, batch_size=10, output_path=output_path)