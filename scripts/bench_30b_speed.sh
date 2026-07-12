#!/bin/bash
# Benchmark Qwen3-30B-A3B MoE decode speed with ExpertOffloader (slice-LRU + 2-GPU split).
# Usage: scripts/bench_30b_speed.sh /path/to/hf_cache
set -e
CKPTDIR="${1:?pass hf_cache path}"
docker run --rm --runtime=nvidia --gpus all --ipc=host --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e DEMO_EXPERT_CACHE_GIB="${DEMO_EXPERT_CACHE_GIB:-11}" \
    -e DEMO_EXPERT_SPLIT_GPU="${DEMO_EXPERT_SPLIT_GPU:-1}" \
    -e DEMO_EXPERT_SLICE_MODE="${DEMO_EXPERT_SLICE_MODE:-1}" \
    -e CKPTDIR="$CKPTDIR" --mount type=bind,src="$CKPTDIR",target="$CKPTDIR" \
    qwen3_l:latest python3 -c "
import os,time,torch
from src.inject_kernel import inject_fp8_kernel
from src.expert_streamer import ExpertOffloader
from src.models import get_model
from transformers import AutoModelForCausalLM, AutoTokenizer
inject_fp8_kernel()
p=get_model('qwen3-30b-a3b').path(os.environ['CKPTDIR'])
m=AutoModelForCausalLM.from_pretrained(p,torch_dtype='auto',device_map='cpu',local_files_only=True,attn_implementation='sdpa').eval()
ExpertOffloader.install(m)
tok=AutoTokenizer.from_pretrained(p,local_files_only=True)
ids=tok('Count from 1 to 8.',return_tensors='pt').input_ids.to('cuda:0')
print('WARMUP',flush=True)
t0=time.time()
with torch.no_grad():
    out=m.generate(ids,max_new_tokens=4,do_sample=False,pad_token_id=tok.eos_token_id)
print('warmup:',round(time.time()-t0,1),'s for 4 tok',flush=True)
print('MEASURE',flush=True)
t0=time.time()
n=12
with torch.no_grad():
    out=m.generate(out,max_new_tokens=n,do_sample=False,pad_token_id=tok.eos_token_id)
el=time.time()-t0
print(f'SPEED_OK {n} tok in {el:.1f}s = {n/el:.2f} tok/s',flush=True)
for g in range(torch.cuda.device_count()):
    u=round(torch.cuda.max_memory_allocated(g)/1024**3,2)
    print(f'GPU{g} peak VRAM: {u} GiB',flush=True)
"