## Training

For model training and fine-tuning, you can directly use the "dynamictrl" environment.

### Data Preparation

#### Folder structure:
```
Training_data/
│
├── videos/
│   ├── 0000001.mp4
│   └── 0000002.mp4
│
├── poses/
│   ├── 0000001.mp4
│   └── 0000001.mp4
│
├── prompts.txt
│
├── video.txt
```

<font color=Coral>Tips:</font> You should use a pose estimation algorithm to extract the human pose, such as the DWPose method. To obtain the prompts for the training video, we select one frame from the video and use Qwen2-VL to understand the image content, including the human's appearance and background details.

#### Data format of video.txt
```
{
  /home/user/data/Traing_data/videos/0000001.mp4
  /home/user/data/Traing_data/videos/0000002.mp4
  ...
}
```

#### Data format of prompts.txt
```
{
  0000001.mp4#####The video descripts xxx. The human xxx. The background xxx.
  0000002.mp4#####The video descripts xxx. The human xxx. The background xxx.
  ...
}
```

### Fine-tuning:

```bash
export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=16
export DS_SKIP_CUDA_CHECK=1

GPU_IDS="0,1,2,3,4,5,6,7"

# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATES=("5e-6")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw")
MAX_TRAIN_STEPS=("50000")
POSE_CONTROL_FUNCTION="padaln"

# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed.yaml"

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.
# This example assumes you downloaded an already prepared dataset from HF CLI as follows:
DATA_ROOT="/home/user/data/Traing_data"
CAPTION_COLUMN="prompts.txt"
VIDEO_COLUMN="video.txt"
MODEL_PATH="./checkpoints/DynamiCtrl"

# Set ` --load_tensors ` to load tensors from disk instead of recomputing the encoder process.
# Launch experiments with different hyperparameters

TRAINING_NAME="training_name"

for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        output_dir="/home/user/exps/train/dynamictrl-sft_${TRAINING_NAME}_${optimizer}_steps_${steps}_lr-schedule_${lr_schedule}_learning-rate_${learning_rate}/"
        
        cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
          --gpu_ids $GPU_IDS --main_process_port 10086 \
          scripts/dynamictrl_sft_finetune.py \
          --pretrained_model_name_or_path  $MODEL_PATH \
          --data_root $DATA_ROOT \
          --caption_column $CAPTION_COLUMN \
          --video_column $VIDEO_COLUMN \
          --id_token BW_STYLE \
          --height_buckets 1360 \
          --width_buckets 768 \
          --height 1360 \
          --width 768 \
          --frame_buckets 37 \
          --dataloader_num_workers 1 \
          --pin_memory \
          --enable_control_pose \
          --pose_control_function $POSE_CONTROL_FUNCTION \
          --validation_prompt \"input the test prompt here.\" \
          --validation_images \"/home/user/data/dynamictrl_train/test.jpg\"
          --validation_driving_videos \"/home/user/data/dynamictrl_train/test_driving_video.mp4\"
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --validation_epochs 100000 \
          --validation_steps 100000 \
          --seed 42 \
          --mixed_precision bf16 \
          --output_dir $output_dir \
          --max_num_frames 37 \
          --train_batch_size 64 \
          --max_train_steps $steps \
          --checkpointing_steps 1000 \
          --gradient_checkpointing \
          --gradient_accumulation_steps 4 \
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
          --lr_warmup_steps 1000 \
          --lr_num_cycles 1 \
          --enable_slicing \
          --enable_tiling \
          --noised_image_dropout 0.05 \
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --max_grad_norm 1.0 \
          --allow_tf32 \
          --report_to tensorboard \
          --nccl_timeout 1800"
        
        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done

```