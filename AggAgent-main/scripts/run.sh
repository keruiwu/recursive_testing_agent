#!/bin/bash
conda activate aggagent
cd /gpfs/u/scratch/MPRG/MPRGhhzz/kerui/recursive_testing_agent/AggAgent-main
export HF_HUB_OFFLINE=1
export HF_LOCAL_DEVICE_MAP=auto
export HF_LOCAL_TORCH_DTYPE=float16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python aggregation/aggregate.py \
  --strategy aggagent \
  --model "hf:/gpfs/u/scratch/MPRG/MPRGhhzz/kerui/models/Qwen3-14B" \
  --judge_llm "hf:/gpfs/u/scratch/MPRG/MPRGhhzz/kerui/models/Qwen3-14B" \
  --task deepsearchqa \
  --k 4 \
  --max_workers 2 \
  --hf_device_map auto \
  --hf_torch_dtype float16 \
  --hf_max_new_tokens 4096 \
  --hf_temperature 0.2 \
  --hf_top_p 0.95 \
  output/rollout/Qwen3.5-122B-A10B/deepsearchqa