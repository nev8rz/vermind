#!/usr/bin/env python3
"""
VLM 训练调试脚本 - 检查 loss 是否真的不下降
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer
from vermind_models import VerMindVLM, VLMConfig
from data_loader.vlm_dataset import VLMDataset, vlm_collate_fn
from torch.utils.data import DataLoader

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else './output/dpo/dpo_768/checkpoint_1610'
    data_path = sys.argv[2] if len(sys.argv) > 2 else './dataset/vlm_pretrain.parquet'
    
    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = VerMindVLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = model.cuda().train()
    
    # 冻结设置
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"可训练参数: {trainable:.2f}M (应该是 ~1.18M)")
    
    # 加载数据
    print(f"\n加载数据: {data_path}")
    dataset = VLMDataset(parquet_path=data_path, tokenizer=tokenizer, max_length=512)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=vlm_collate_fn)
    
    # 优化器 - 使用较大学习率测试
    optimizer = torch.optim.AdamW(model.vision_proj.parameters(), lr=1e-3)
    
    print("\n开始训练 20 步...")
    losses = []
    
    for step, batch in enumerate(loader):
        if step >= 20:
            break
        
        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()
        pixel_values = batch["pixel_values"].cuda()
        
        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=labels, pixel_values=pixel_values)
        loss = output.loss
        
        loss.backward()
        
        # 记录梯度范数
        grad_norm = model.vision_proj.mlp[0].weight.grad.norm().item()
        
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 5 == 0:
            print(f"Step {step}: loss={loss.item():.4f}, grad_norm={grad_norm:.6f}")
    
    print(f"\n=== 结果 ===")
    print(f"初始 loss: {losses[0]:.4f}")
    print(f"最终 loss: {losses[-1]:.4f}")
    print(f"下降: {losses[0] - losses[-1]:.4f}")
    
    if losses[0] - losses[-1] < 0.1:
        print("\n⚠️  Warning: Loss 下降不明显！")
        print("可能原因:")
        print("1. 数据中没有 image_ids (检查 VLMDataset 是否正确替换 <image>)")
        print("2. 学习率太小 (尝试增大到 1e-3)")
        print("3. 数据质量有问题")
    else:
        print("\n✅ Loss 正常下降！")

if __name__ == "__main__":
    main()
