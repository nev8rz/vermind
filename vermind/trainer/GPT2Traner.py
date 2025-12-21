import math
import os
import pickle
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..configs import GPT2Config
from ..models import GPT2


@dataclass
class TrainArgs:
    # I/O
    out_dir: str = "out"
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True

    # init: "scratch" | "resume" | "gpt2" | "gpt2-medium" | "gpt2-large" | "gpt2-xl"
    init_from: str = "scratch"

    # data
    dataset: str = "openwebtext"
    batch_size: int = 12
    block_size: int = 1024
    gradient_accumulation_steps: int = 40

    # model (scratch 时用)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

    # optimizer
    learning_rate: float = 6e-4
    max_iters: int = 600_000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # lr schedule
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5

    # system
    device: str = "cuda"
    dtype: str = "bfloat16"  # float32 | float16 | bfloat16
    compile: bool = True
    backend: str = "nccl"

    use_model_configure_optimizers: bool = True

    # -------------------------
    # SwanLab（新增）
    # -------------------------
    swanlab: bool = True
    swanlab_project: str = "gpt2"
    swanlab_workspace: str | None = None  # 组织空间（可选），不填就是个人空间 :contentReference[oaicite:3]{index=3}
    swanlab_experiment_name: str | None = None  # 不填会自动生成 :contentReference[oaicite:4]{index=4}
    swanlab_mode: str | None = None  # "cloud" | "offline" | "local" | "disabled"（可选） :contentReference[oaicite:5]{index=5}
    swanlab_logdir: str | None = None  # 默认 swanlog（可选） :contentReference[oaicite:6]{index=6}

    # smoothing
    loss_ema_beta: float = 0.98  # EMA 平滑系数，越大越平滑
    custom_chart_window: int = 500  # custom chart 只画最近 N 个点（避免太大）


class GPT2Trainer:
    def __init__(self, cfg: TrainArgs):
        self.cfg = cfg

        # DDP state
        self.ddp = False
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.master = True
        self.device = cfg.device
        self.grad_accum = cfg.gradient_accumulation_steps

        # train state
        self.iter_num = 0
        self.best_val_loss = 1e9

        # SwanLab state（新增）
        self.swan_run = None
        self.swan_run_id = None
        self._train_loss_ema = None
        self._val_loss_ema = None
        self._loss_hist_step = []
        self._loss_hist_raw = []
        self._loss_hist_ema = []

        # setup ddp first
        self._setup_ddp()

        if self.master:
            os.makedirs(cfg.out_dir, exist_ok=True)

        # seed & tf32
        torch.manual_seed(1337 + self.rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # autocast
        self.device_type = "cuda" if "cuda" in self.device else "cpu"
        self.ptdtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[cfg.dtype]
        self.ctx = nullcontext() if self.device_type == "cpu" else torch.amp.autocast(
            device_type=self.device_type, dtype=self.ptdtype
        )

        # fp16 scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == "float16"))

        # data dir & vocab
        self.data_dir = os.path.join("data", cfg.dataset)
        self.meta_vocab_size = self._load_meta_vocab_size()

        # build model & optimizer (includes scratch/resume/gpt2*)
        self.model, self.raw_model, self.optimizer = self._build_model_and_optimizer()

        # compile
        if cfg.compile:
            self.model = torch.compile(self.model)

        # DDP wrap
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        # SwanLab init（新增）：只在 master 初始化，DDP 安全
        if self.master and cfg.swanlab:
            self._init_swanlab()

        # tokens per iter
        tokens_per_iter = self.grad_accum * self.world_size * cfg.batch_size * cfg.block_size
        if self.master:
            print(f"tokens per iteration will be: {tokens_per_iter:,}")

    # -------------------------
    # DDP
    # -------------------------
    def _setup_ddp(self):
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if not self.ddp:
            return

        dist.init_process_group(backend=self.cfg.backend)

        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        self.device = f"cuda:{self.local_rank}"
        torch.cuda.set_device(self.device)

        self.master = self.rank == 0

        assert self.cfg.gradient_accumulation_steps % self.world_size == 0, \
            "gradient_accumulation_steps must be divisible by world_size"
        self.grad_accum = self.cfg.gradient_accumulation_steps // self.world_size

    def _cleanup_ddp(self):
        if self.ddp:
            dist.destroy_process_group()

    # -------------------------
    # SwanLab（新增）
    # -------------------------
    def _init_swanlab(self):
        """
        目标：
        1) DDP 下只在 master 初始化
        2) resume 时恢复到同一个 swanlab experiment（用 id + resume）:contentReference[oaicite:7]{index=7}
        """
        import swanlab

        cfg = self.cfg

        # resume：从 checkpoint 里拿 swanlab_run_id（在 _build_model_and_optimizer 里会填）
        # 这样你只要 init_from="resume" 就能把曲线接上，不会新开实验。
        init_kwargs = dict(
            project=cfg.swanlab_project,
            workspace=cfg.swanlab_workspace,
            experiment_name=cfg.swanlab_experiment_name,
            config=asdict(cfg),
        )
        # mode / logdir 可选
        if cfg.swanlab_mode is not None:
            init_kwargs["mode"] = cfg.swanlab_mode
        if cfg.swanlab_logdir is not None:
            init_kwargs["logdir"] = cfg.swanlab_logdir

        # 如果是 resume 且我们有 run_id：恢复同一个实验 :contentReference[oaicite:8]{index=8}
        if cfg.init_from == "resume" and self.swan_run_id:
            init_kwargs["resume"] = "must"  # 必须存在，否则抛错（更安全）
            init_kwargs["id"] = self.swan_run_id
        else:
            # 非 resume：默认开新实验（或你也可以改成 resume="allow"）
            # init_kwargs["resume"] = "allow"
            pass

        self.swan_run = swanlab.init(**init_kwargs)
        self.swan_run_id = getattr(self.swan_run, "id", None)

        # 把 run_id 写到本地文件（可读性好，排查方便）
        # 注意：真正 resume 还是以 ckpt 里的 swanlab_run_id 为准
        if self.swan_run_id:
            with open(os.path.join(cfg.out_dir, "swanlab_run_id.txt"), "w", encoding="utf-8") as f:
                f.write(self.swan_run_id)

        print(f"[SwanLab] enabled. run_id={self.swan_run_id}")

    def _swan_log(self, data: dict, step: int):
        """
        显式指定 step，确保和 iter_num 对齐，resume 后曲线不断档 :contentReference[oaicite:9]{index=9}
        """
        if not (self.master and self.cfg.swanlab and self.swan_run):
            return
        import swanlab
        swanlab.log(data, step=step)

    def _ema_update(self, prev_ema: float | None, value: float, beta: float) -> float:
        if prev_ema is None:
            return float(value)
        return beta * float(prev_ema) + (1.0 - beta) * float(value)

    def _log_custom_loss_chart(self):
        """
        custom chart：用 swanlab.echarts.Line 画最近 N 点 raw vs ema :contentReference[oaicite:10]{index=10}
        频率：我们只在 eval 时发一次，避免太频繁/太大。
        """
        if not (self.master and self.cfg.swanlab and self.swan_run):
            return
        if len(self._loss_hist_step) < 2:
            return

        import swanlab

        win = int(self.cfg.custom_chart_window)
        xs = self._loss_hist_step[-win:]
        ys_raw = self._loss_hist_raw[-win:]
        ys_ema = self._loss_hist_ema[-win:]

        # swanlab.echarts 兼容 pyecharts 风格 :contentReference[oaicite:11]{index=11}
        line = swanlab.echarts.Line()
        line.add_xaxis([str(x) for x in xs])
        line.add_yaxis("train_loss_raw", ys_raw)
        line.add_yaxis("train_loss_ema", ys_ema)

        # chart 作为一个“媒体对象”记录到 swanlab
        self._swan_log({"charts/train_loss_raw_vs_ema": line}, step=self.iter_num)

    # -------------------------
    # Data
    # -------------------------
    def _load_meta_vocab_size(self):
        meta_path = os.path.join(self.data_dir, "meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            if self.master:
                print(f"found vocab_size = {meta['vocab_size']} (inside {meta_path})")
            return meta["vocab_size"]
        return None

    def get_batch(self, split: str):
        cfg = self.cfg
        bin_path = os.path.join(self.data_dir, "train.bin" if split == "train" else "val.bin")
        data = np.memmap(bin_path, dtype=np.uint16, mode="r")

        ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + cfg.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + cfg.block_size]).astype(np.int64)) for i in ix])

        if self.device_type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y

    def _build_adamw(self, model: torch.nn.Module):
        cfg = self.cfg

        param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        fused = False
        try:
            import inspect as _inspect
            fused_available = "fused" in _inspect.signature(torch.optim.AdamW).parameters
            fused = fused_available and (self.device_type == "cuda")
        except Exception:
            fused = False

        extra_args = {"fused": True} if fused else {}
        opt = torch.optim.AdamW(
            optim_groups,
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            **extra_args,
        )

        if self.master:
            num_decay = sum(p.numel() for p in decay_params)
            num_nodecay = sum(p.numel() for p in nodecay_params)
            print(f"AdamW groups: decay={num_decay:,} params, nodecay={num_nodecay:,} params, fused={fused}")
        return opt

    # -------------------------
    # Model / Optim / init_from (scratch/resume/gpt2*)
    # -------------------------
    def _build_model_and_optimizer(self):
        cfg = self.cfg

        # ---- init_from: gpt2* (pretrain) ----
        if cfg.init_from.startswith("gpt2"):
            if self.master:
                print(f"Initializing from OpenAI GPT-2 weights: {cfg.init_from}")
            override_args = {"dropout": cfg.dropout}
            model = GPT2.from_pretrained(cfg.init_from, override_args=override_args).to(self.device)

            # 如需裁剪 block_size（你想训练更短上下文）
            if cfg.block_size < model.config.block_size:
                model.crop_block_size(cfg.block_size)

            raw_model = model

            # optimizer
            if cfg.use_model_configure_optimizers:
                optimizer = raw_model.configure_optimizers(
                    cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), self.device_type
                )
            else:
                optimizer = self._build_adamw(raw_model)

            self.iter_num = 0
            self.best_val_loss = 1e9

            # SwanLab：pretrain 新实验，不做 resume
            self.swan_run_id = None
            return model, raw_model, optimizer

        # ---- init_from: resume ----
        checkpoint = None
        if cfg.init_from == "resume":
            ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"resume requested but checkpoint not found: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            if self.master:
                print(f"Resuming training from {cfg.out_dir}")

            # SwanLab：从 checkpoint 里恢复 run_id（用于 swanlab.init(resume=..., id=...)）:contentReference[oaicite:12]{index=12}
            self.swan_run_id = checkpoint.get("swanlab_run_id", None)

        # ---- init_from: scratch / resume -> build config ----
        vocab_size = self.meta_vocab_size if self.meta_vocab_size is not None else 50304
        model_args = dict(
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            block_size=cfg.block_size,
            bias=cfg.bias,
            vocab_size=vocab_size,
            dropout=cfg.dropout,
        )

        # resume: 强制对齐 ckpt 中 model_args（关键）
        if checkpoint is not None:
            ckpt_model_args = checkpoint["model_args"]
            for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
                model_args[k] = ckpt_model_args[k]

        # create model
        config = GPT2Config(**model_args)
        model = GPT2(config).to(self.device)
        raw_model = model

        # optimizer
        if cfg.use_model_configure_optimizers:
            optimizer = raw_model.configure_optimizers(
                cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), self.device_type
            )
        else:
            optimizer = self._build_adamw(raw_model)

        # load resume
        if checkpoint is not None:
            sd = checkpoint["model"]
            unwanted_prefix = "_orig_mod."
            for k in list(sd.keys()):
                if k.startswith(unwanted_prefix):
                    sd[k[len(unwanted_prefix):]] = sd.pop(k)
            raw_model.load_state_dict(sd)
            optimizer.load_state_dict(checkpoint["optimizer"])
            self.iter_num = checkpoint["iter_num"]
            self.best_val_loss = checkpoint["best_val_loss"]
            if self.master:
                print(f"Resumed: iter={self.iter_num}, best_val_loss={self.best_val_loss:.4f}")

            # EMA/历史也可恢复（可选）
            self._train_loss_ema = checkpoint.get("train_loss_ema", None)
            self._val_loss_ema = checkpoint.get("val_loss_ema", None)

        return model, raw_model, optimizer

    # -------------------------
    # LR schedule
    # -------------------------
    def get_lr(self, it: int):
        cfg = self.cfg
        if it < cfg.warmup_iters:
            return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
        if it > cfg.lr_decay_iters:
            return cfg.min_lr
        decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
        decay_ratio = max(0.0, min(1.0, decay_ratio))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    # -------------------------
    # Eval
    # -------------------------
    @torch.no_grad()
    def estimate_loss(self):
        cfg = self.cfg
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(cfg.eval_iters, device="cpu")
            for k in range(cfg.eval_iters):
                X, Y = self.get_batch(split)
                with self.ctx:
                    _, loss = self.model(X, Y)
                losses[k] = loss.detach().float().cpu().item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    # -------------------------
    # Checkpoint
    # -------------------------
    def save_checkpoint(self):
        cfg = self.cfg
        raw = self.model.module if (self.ddp and isinstance(self.model, DDP)) else self.model

        ckpt = {
            "model": raw.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "model_args": {
                "n_layer": raw.config.n_layer,
                "n_head": raw.config.n_head,
                "n_embd": raw.config.n_embd,
                "block_size": raw.config.block_size,
                "bias": raw.config.bias,
                "vocab_size": raw.config.vocab_size,
                "dropout": raw.config.dropout,
            },
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "config": asdict(cfg),

            # SwanLab：保存 run_id，保证 resume 能续写同一实验 :contentReference[oaicite:13]{index=13}
            "swanlab_run_id": self.swan_run_id,

            # smoothing：保存 EMA 状态（可选，但很实用）
            "train_loss_ema": self._train_loss_ema,
            "val_loss_ema": self._val_loss_ema,
        }
        path = os.path.join(cfg.out_dir, "ckpt.pt")
        torch.save(ckpt, path)
        if self.master:
            print(f"saving checkpoint to {cfg.out_dir}")

    # -------------------------
    # Train loop
    # -------------------------
    def train(self):
        cfg = self.cfg

        X, Y = self.get_batch("train")
        t0 = time.time()
        local_iter_num = 0
        running_mfu = -1.0

        while True:
            # lr
            lr = self.get_lr(self.iter_num) if cfg.decay_lr else cfg.learning_rate
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # eval + ckpt
            if self.iter_num % cfg.eval_interval == 0 and self.master:
                losses = self.estimate_loss()
                train_loss = float(losses["train"])
                val_loss = float(losses["val"])

                # smoothing：eval 的 EMA（单独一条曲线）
                self._val_loss_ema = self._ema_update(self._val_loss_ema, val_loss, cfg.loss_ema_beta)

                print(
                    f"step {self.iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}"
                )

                # SwanLab：显式 step 对齐 iter_num :contentReference[oaicite:14]{index=14}
                self._swan_log(
                    {
                        "eval/train_loss": train_loss,
                        "eval/val_loss": val_loss,
                        "eval/val_loss_ema": float(self._val_loss_ema),
                        "lr": lr,
                        "iter": self.iter_num,
                    },
                    step=self.iter_num,
                )

                # custom chart：只在 eval 时更新一次
                self._log_custom_loss_chart()

                if (val_loss < self.best_val_loss) or cfg.always_save_checkpoint:
                    self.best_val_loss = min(self.best_val_loss, val_loss)
                    if self.iter_num > 0:
                        self.save_checkpoint()

            if self.iter_num == 0 and cfg.eval_only:
                break

            # grad accumulation
            for micro_step in range(self.grad_accum):
                if self.ddp:
                    self.model.require_backward_grad_sync = (micro_step == self.grad_accum - 1)

                with self.ctx:
                    _, loss = self.model(X, Y)
                    loss = loss / self.grad_accum

                # prefetch next batch
                X, Y = self.get_batch("train")

                self.scaler.scale(loss).backward()

            # clip grad
            if cfg.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            # step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # log
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if self.iter_num % cfg.log_interval == 0 and self.master:
                lossf = loss.detach().float().cpu().item() * self.grad_accum

                # smoothing：train 的 EMA
                self._train_loss_ema = self._ema_update(self._train_loss_ema, lossf, cfg.loss_ema_beta)

                # 维护历史（给 custom chart 用）
                self._loss_hist_step.append(int(self.iter_num))
                self._loss_hist_raw.append(float(lossf))
                self._loss_hist_ema.append(float(self._train_loss_ema))

                # MFU
                mfu_str = ""
                raw = self.model.module if (self.ddp and isinstance(self.model, DDP)) else self.model
                if hasattr(raw, "estimate_mfu") and local_iter_num >= 5:
                    mfu = raw.estimate_mfu(cfg.batch_size * self.grad_accum, dt)
                    running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu
                    mfu_str = f", mfu {running_mfu*100:.2f}%"

                print(f"iter {self.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms{mfu_str}")

                # SwanLab：显式 step，记录 raw + ema + time + mfu + lr :contentReference[oaicite:15]{index=15}
                log_payload = {
                    "train/loss": float(lossf),
                    "train/loss_ema": float(self._train_loss_ema),
                    "time/ms": float(dt * 1000.0),
                    "lr": lr,
                    "iter": self.iter_num,
                }
                if running_mfu >= 0:
                    log_payload["mfu"] = float(running_mfu)

                self._swan_log(log_payload, step=self.iter_num)

            self.iter_num += 1
            local_iter_num += 1

            if self.iter_num > cfg.max_iters:
                break

        # SwanLab：正常结束（一般会自动 finish；显式调用也更稳）
        if self.master and self.cfg.swanlab and self.swan_run:
            try:
                self.swan_run.finish()
            except Exception:
                pass

        self._cleanup_ddp()
