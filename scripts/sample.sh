export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ckpt=${1:-checkpoints/gat_b2_256.pt}

torchrun --standalone --nproc_per_node=1 generate.py \
  --ckpt="${ckpt}" \
  --num-fid-samples=50000 \
  --per-proc-batch-size=32
