

python little_nurse/finetune.py \
    --model_path "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/Checkpoints/Llama-2-7b-chat-hf" \
    --data_path "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/PaddleNLP/Data/datasets/aiwei.json" \
    --output_dir "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/PaddleNLP/outputs/aiwei_lora_64" \
    --lora_enable "True" \
    --lora_r "32" \
    --lora_alpha "64" \
    --bf16 "True" \
    --tf32 "True" \
    --per_device_train_batch_size "4" \
    --gradient_accumulation_steps "1" \
    --per_device_eval_batch_size "1" \
    --num_train_epochs "1" \
    --learning_rate "1e-4" \
    --warmup_ratio "0.03" \
    --lr_scheduler_type "cosine" \
    --logging_steps "1" \
    --save_strategy "epoch" \
    --eval_steps "100" \
    --save_total_limit "3"
    