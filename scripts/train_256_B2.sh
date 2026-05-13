export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

model="GAT-B/2"
modelD="GAT-B/2"

wandb_name="GAT"
resolution=256
batch_size=512


data_path="../dataset"
expdir="../exps"


expname="gat_b2_256"

resume_step=0
R1_every=1
R1_gamma=1e-1
R2_gamma=1e-1
learning_rate=2e-4
enc_type=dinov2-vit-b

model_cleaned=${model//\//-}
modelD_cleaned=${modelD//\//-}
expname="G-${model_cleaned}_D-${modelD_cleaned}_${expname}"


accelerate launch --main_process_port 29501 train.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="bf16" \
  --seed=0 \
  --sampling-steps=1250 \
  --resolution=${resolution} \
  --model=${model} \
  --modelD=${modelD} \
  --enc-type="${enc_type}" \
  --proj-coeff=1.0 \
  --output-dir=${expdir} \
  --exp-name="${expname}" \
  --batch-size=${batch_size} \
  --data-dir="${data_path}" \
  --resume-step=${resume_step} \
  --wandb-name="${wandb_name}" \
  --learning-rate=${learning_rate} \
  --R1_gamma=${R1_gamma} \
  --R2_gamma=${R2_gamma} \
  --R1_every=${R1_every} \
  --R2_every=${R1_every}
