import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from train_vocab import Task

from config_util import (
    get_params,
    setup_env,
    setup_model_and_optimizer
)
from config import PARAMS_CONFIG

def log_tensorboard_metrics(writer, losses, lr, running_mfu, iter_num):                                                                                                                                                                    │
    """Helper function for tensorboard metric writes"""

    # This places train and val on the same graph
    writer.add_scalars(                                                                                                                                                                                                                    │
        "loss", { "train": losses['train'], "val": losses['val'] }, iter_num                                                                                                                                                               │
        )                                                                                                                                                                                                                                  │
    # TODO: add complete
    writer.add_scalar("mfu_pct", running_mfu * 100, iter_num)                                                                                                                                                                              │
    writer.add_scalar("lr", lr, iter_num)                                                                                                                                                                                                  │

@torch.no_grad()
def estimate_loss(model, eval_iters, iter_batches, ctx):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out

def get_lr(iter_num, optimizer_params):
    # 1) linear warmup for warmup_iters steps
    if iter_num < optimizer_params['warmup_iters']:
        return optimizer_params['learning_rate'] * iter_num / optimizer_params['warmup_iters']
    # 2) if iter_num > decay_lr_iters, return min learning rate
    if iter_num > optimizer_params['decay_lr_iters']:
        return optimizer_params['min_lr']
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter_num - optimizer_params['warmup_iters']) / (optimizer_params['decay_lr_iters'] - optimizer_params['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return optimizer_params['min_lr'] + coeff * (optimizer_params['learning_rate'] - optimizer_params['min_lr'])

def train_launcher(env_params, vocab_params, model_params, optimizer_params, trainer_params, wandb_params):
    # 0. Setup global enviroment and log tool (tensorboard and wandb) for visualization
    setup_env(env_params)
    device, dtype = env_params['device'], env_params['dtype']

    board_writer = SummaryWriter() # Tensor Borad Writer

    if wandb_params['wandb_log']:
        import wandb
        wandb.init(project=wandb_params['wandb_project'], name=wandb_params['wandb_run_name'])

    # 1. Check multi-core training. But for now, we only have one GPU on glcoud
    batch_size = trainer_params['batch_size']
    gradient_accumulation_steps = trainer_params['gradient_accumulation_steps']
    max_seq_len = model_params['max_seq_len']

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len

    if master_process:
        #print(f"tokens per iteration will be: {tokens_per_iter:,}")
        #print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")
        os.makedirs(env_params['model_dir'], exist_ok=True)

    # 2. Setup Pytorch mix-precision training (FP32 + BF16)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    # 3. Initialize the dataset loader
    iter_batches = partial(
        Task.iter_batches,
        data_path = vocab_params['vocab_data'],
        vocab_size = vocab_params['vocab_size'],
        vocab_source = vocab_params['vocab_source'],
        batch_size = batch_size,
        max_seq_len = max_seq_len,
        device = device,
        num_workers = 0
    )

    # 4. Initilize or resume model and optimizer
    model, optimizer, model_args, iter_num, best_val_loss = setup_model_and_optimizer(
        model_params = model_params,
        optimizer_params = optimizer_params,
        init_from = trainer_params['init_from'],
        ckpt_path = os.path.join(env_params['model_dir'], env_params['ckpt']),
        vocab_size = vocab_params['vocab_size'],
        device = device
    )

    if ddp:
        # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
        # construction time since NCCL does not support `ComplexFloat`
        model._ddp_params_and_buffers_to_ignore = {'freqs_cis'}
        model = DDP(model, device_ids=[ddp_local_rank])

    # 5. Most Important !!!. Start training loop.
    # Fetch data batch
    train_batch_iter = iter_batches(split='train')
    X, Y = next(train_batch_iter)
    # number of iterations in the life time of the current process
    # same as iter_num when training a model from scratch
    local_iter_num = 0
    # unwrap DDP container if needed
    raw_model = model.module if ddp else model
    # GPU utilization monitor
    running_mfu = -1.0

    # Start training loop
    t0 = time.time()
    while True:
        # determine and set learning rate for this iteration
        if optimizer_params['decay_lr']:
            lr = get_lr(iter_num, optimizer_params)
        else:
            lr = optimizer_params['learning_rate']
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % trainer_params['eval_interval'] == 0 and master_process:
            train_val_loss = estimate_loss(model, trainer_params['eval_iters'], iter_batches, ctx)
            '''
            # Start Evaluation
            train_val_loss = {}
            model.eval()
            for split in ['train', 'val']:
                batch_iter = iter_batches(split=split)
                iter_loss = torch.zeros(trainer_params['eval_iters'])
                for k in range(trainer_params['eval_iters']):
                    X, Y = next(batch_iter)
                    with ctx:
                        logits, loss = model(X, Y)
                    iter_loss[k] = loss.item()
                train_val_loss[split]=iter_loss.mean()
            '''
            model.train()

            print(f"step {iter_num}: train loss {train_val_loss['train']:.4f}, train ppl {math.exp(train_val_loss['train']):.4f}, val loss {train_val_loss['val']:.4f}, val ppl {math.exp(train_val_loss['val']):.4f}")
            if wandb_params['wandb_log']:
                try:
                    wandb.log({'iter': iter_num, 'tokens': iter_num * tokens_per_iter, 'loss/train': train_val_loss['train'], 'loss/val': train_val_loss['val'], 'lr': lr, 'mfu': running_mfu * 100,})
                except Exception as e:
                    print(f'logging to wandb failed: {e}')

            if train_val_loss['val'] < best_val_loss or trainer_params['always_save_ckpt']:
                best_val_loss = train_val_loss['val']
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                    }
                    model_path = env_params['model_dir'] + '/' + str(model_params['n_layers']) + '_' +str(model_params['n_heads']) +  '_' + \
                                 str(model_params['n_embd']) + '_' + env_params['ckpt']
                    print(f"saving checkpoint to {model_path}")
                    torch.save(checkpoint, model_path)
                    #print(f"saving checkpoint to {os.path.join(env_params['model_dir'], env_params['ckpt'])}")
                    #torch.save(checkpoint, os.path.join(env_params['model_dir'], env_params['ckpt']))
        if trainer_params['eval_only']:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
            with ctx:
                logits, loss = model(X, Y)
                # loss divided account for the fact that gradients are accumulated
                # over multiple steps before the optimization step.
                # pytoroch will accumulate the gradient automatically
                loss = loss / gradient_accumulation_steps
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = next(train_batch_iter)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

        # clip the gradient before updating the weight
        if optimizer_params['grad_clip'] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_params['grad_clip'])

        # updata the weight with optimizer step
        # scaler if training in bf16
        scaler.step(optimizer)
        scaler.update()

        # flush the gradient for the next weight updating
        optimizer.zero_grad(set_to_none=True)

        # training information print
        t1 = time.time()
        dt = t1 -t0
        t0 = t1
        if iter_num % trainer_params['log_interval'] == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            # calculate the GPU ultization after settling training loop a bit
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            # print
            #print(f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}% | layers {model_params['n_layers']} | heads {model_params['n_heads']} | embd {model_params['n_embd']}")
            # tensorboard record
            board_writer.add_scalar("Loss/train", lossf, iter_num)
        iter_num += 1
        local_iter_num += 1

        # termination training if reaching the maxium iteration
        if iter_num > trainer_params['max_iters']:
            break

    # Flush tensorboard record to disk
    board_writer.flush()
    board_writer.close()

    if ddp:
        destroy_process_group()

    print('Training Ready')

    return best_val_loss

if __name__ == '__main__':
    #print(get_params(params_config=PARAMS_CONFIG))
    train_launcher(**get_params(params_config=PARAMS_CONFIG))
