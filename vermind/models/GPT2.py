import inspect
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..configs import GPT2Config


class LayerNorm(nn.Module):
    '''
    对单个样本的所有特征进行归一化，而不是跨越 Batch 进行归一化
    $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
    $$y_i = \gamma \hat{x}_i + \beta$$
    其中 \gamma 和 \beta 是可学习的参数
    '''
    
    def __init__(self,hidden_size,eps = 1e-5,bias = True):
        # 不用bias，更加稳定（PaLM 论文）
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size)) # gamma
        if bias: # beta
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter('bias',None)
        
    def forward(self,x):
        '''
        x [...,hidden_size]
        '''
        x_type = x.dtype
        x = x.float() 
        # 计算均值和方差
        mean = x.mean(dim=-1,keepdim=True)
        variance = x.var(dim=-1,keepdim=True,unbiased=False)
        
        # 归一化,使用rsqrt
        x_hat = (x - mean) * torch.rsqrt(variance + self.eps)
        
        x_hat = x_hat.to(x_type)
        
        if self.bias is not None:
            return self.weight * x_hat + self.bias
        return self.weight * x_hat
    
class CausalSelfAttention(nn.Module): 
    '''
    因果自注意力机制，mask掉未来的信息，即上三角矩阵部分
    tips: gpt2 实现 使用c_attn 来同时计算 q,k,v，然后再拆分,c_proj 来做线性变换
    '''
    
    def __init__(self,config):
        super().__init__()
        # 确保可以均匀划分头数
        assert config.n_embd % config.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd,3*config.n_embd,bias = config.bias)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd,bias = config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.dropout = config.dropout
        
        self.flash = hasattr(torch.nn.functional,'scaled_dot_product_attention')
        
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch 2.0+ and a compatible GPU.")
            self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))
        
    
    def forward(self,x):
        B,T,C = x.size() # batch size,时间步长，嵌入维度
        
        q,k,v = self.c_attn(x).split(self.n_embd,dim=2)
        
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # B,nh,T,hs
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # B,nh,T,hs
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # B,nh,T,hs
        
        if self.flash:
            attn_output = F.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=self.dropout if self.training else 0,is_causal=True)
        else:
            attn = (q @ k.transpose(-2,-1)) * math.sqrt(k.size(-1)) # B,nh,T,T
            attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0,float('-inf')) # mask掉未来的信息,填充 -inf，softmax 后变为0
            attn = F.softmax(attn,dim=-1)
            # 随机丢取一些看到的词，防止过拟合
            attn = self.attn_dropout(attn)
            attn_output = attn @ v # B,nh,T,hs
            
        attn_output = attn_output.transpose(1,2).contiguous().view(B,T,C) # B,T,C
        # 随机丢取一些特征，防止过拟合
        return self.resid_dropout(self.c_proj(attn_output))
    
class MLP(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd,4 * config.n_embd,bias = config.bias)
        self.gelu = nn.GELU() # GPT-2 使用 gelu 激活函数
        self.c_proj = nn.Linear(4 * config.n_embd,config.n_embd,bias = config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self,x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd,eps=1e-5,bias = config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd,eps=1e-5,bias = config.bias)
        self.mlp = MLP(config)
        
    def forward(self,x):
        # 
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class GPT2(nn.Module):
    
    def __init__(self,config:GPT2Config):
        super().__init__()
        
        assert config.vocab_size is not None
        assert config.block_size is not None
        
        self.config = config
        
        self.transformer = nn.ModuleDict(
            wte = nn.Embedding(config.vocab_size,config.n_embd), # token embedding
            wpe = nn.Embedding(config.block_size,config.n_embd), # position embedding
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # transformer blocks
            ln_f = LayerNorm(config.n_embd,eps=1e-5,bias = config.bias) # final layer norm
        )
        
        
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias = False)
        
        # 指向同一内存，权重共享
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
        for pn,p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p,mean=0.0,std=0.02/math.sqrt(2 * config.n_layer))
                
    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            
    def get_num_params(self,non_embedding = True):
        n_params = sum(p.numel() for p in self.parameters())
        # 减去嵌入层参数
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params
    
    def forward(self,idx,targets = None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, "Cannot forward, model block size is exhausted."
        
        pos = torch.arange(0,t,dtype=torch.long,device=device).unsqueeze(0) # 位置编码
        
        tok_emb = self.transformer.wte(idx) # token embedding
        pos_emb = self.transformer.wpe(pos) # position embedding
        
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        
        # Final layer norm
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            # 计算损失，小概率大惩罚，大概率小惩罚，ignore index -1 表示不计算该位置的损失
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=-1)
        else:
            loss = None
        return logits, loss
    
    def crop_block_size(self, block_size):
        """
        Crop the model to a different (smaller) block size.
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size

        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        override_args = override_args or {}
        assert all(k == "dropout" for k in override_args)

        import torch
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768, dropout=0.1),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024, dropout=0.1),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280, dropout=0.1),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600, dropout=0.1),
        }[model_type]

        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True

        if "dropout" in override_args:
            config_args["dropout"] = override_args["dropout"]

        
        config = GPT2Config(**config_args)
        model = GPT2(config)

        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        gpt2_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = gpt2_hf.state_dict()
        sd_keys_hf = [
            k for k in sd_hf.keys()
            if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias")
        ]

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        transposed_suffixes = (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        )

        for k in sd_keys_hf:
            if k not in sd:
                raise KeyError(f"key {k} not found in your model state_dict")

            if k.endswith(transposed_suffixes):
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"shape mismatch for {k}: hf {sd_hf[k].shape} vs ours {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, f"shape mismatch for {k}: hf {sd_hf[k].shape} vs ours {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        model.load_state_dict(sd)
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 1. 获取模型中所有参数（参数名 -> 参数张量）
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 2. 过滤掉不需要计算梯度的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # 3. 按是否使用 weight decay 对参数进行分组
        #    经验规则：
        #    - 维度 >= 2 的参数（如 Linear / Conv / Embedding 的权重）使用 weight decay
        #    - 维度 < 2 的参数（如 bias、LayerNorm 权重）不使用 weight decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        # 4. 构造优化器的参数组
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # 5. 统计参数数量（用于调试和确认分组是否正确）
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"使用 weight decay 的参数张量数量: {len(decay_params)}, 参数总数: {num_decay_params:,}")
        print(f"不使用 weight decay 的参数张量数量: {len(nodecay_params)}, 参数总数: {num_nodecay_params:,}")

        # 6. 创建 AdamW 优化器，并在条件允许时使用 fused 版本（CUDA 加速）
        #    检查当前 PyTorch 的 AdamW 是否支持 fused 参数
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        #    只有在 CUDA 设备上才启用 fused AdamW
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args
        )

        print(f"是否使用 fused AdamW: {use_fused}")

        # 7. 返回优化器
        return optimizer
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """估算模型 FLOPs 利用率（MFU），相对于 A100 bfloat16 峰值算力"""

        # 模型参数量
        N = self.get_num_params()

        # 模型结构参数
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
    

        # 每个 token 的 FLOPs 估算
        flops_per_token = 6 * N + 12 * L * H * Q * T

        # 每次 forward+backward 的 FLOPs
        flops_per_fwdbwd = flops_per_token * T

        # 每个 iteration 的 FLOPs（考虑梯度累积）
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # 实际 FLOPs 吞吐（每秒）
        flops_achieved = flops_per_iter / dt

        # A100 bfloat16 理论峰值算力
        flops_promised = 312e12

        # 模型 FLOPs 利用率
        return flops_achieved / flops_promised

    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        给定输入 token 序列 idx，逐步生成 max_new_tokens 个新 token
        idx: (batch, seq_len)
        """
        for _ in range(max_new_tokens):
            # 超过最大上下文长度则截断
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # 前向计算，得到 logits
            logits, _ = self(idx_cond)

            # 取最后一个位置的 logits，并做 temperature 缩放
            logits = logits[:, -1, :] / temperature

            # 只保留 top-k 概率最大的 token
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            # softmax 得到概率分布
            probs = F.softmax(logits, dim=-1)

            # 按概率采样下一个 token
            idx_next = torch.multinomial(probs, num_samples=1)

            # 拼接到已有序列，继续生成
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
