# Emo-Llama
Llama2 finetuning: Based on transformers library, finetuning llama2 mental health expert model with lora
llama2微调实战：基于transformers库，加lora微调llama2心理健康专家模型
基座模型：[Llama2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
lora微调权重：
- 艾薇医生 [aiwei](https://huggingface.co/JobsWei/aiwei-lora-llama2-7b-chat-hf)

## 项目简介
本项目基于paddle项目-[使用PaddleNLP 从0构建一个属于你自己的心理大模型](https://aistudio.baidu.com/projectdetail/8002289?channelType=0&channel=0)。由于paddle库实在是难用，而且他那个库finetune好像还存在bug，于是我做了一个基于transformers库和peft微调版本，正好也学习一下微调流程。
* 最低显存占用：20G
## 环境配置
```shell
conda create -n llama python=3.10
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flash-attn --no-build-isolation --use-pep517
```
## Deploy
下载基座模型[llama2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)和lora权重[aiwei](https://huggingface.co/JobsWei/aiwei-lora-llama2-7b-chat-hf)
```shell
python demo/cli.py --model_path $YOUR_LLAMA2_PATH --lora_path $YOUR_LORA_PATH
```
## Dataset
数据集下载链接[emollm dataset](https://aistudio.baidu.com/datasetdetail/276450)
也可以参考[文心一言数据生成流程](https://aistudio.baidu.com/projectdetail/8002289?channelType=0&channel=0)进行自己的数据生成
基座模型llama2下载链接：[huggingface llama2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
## Finetune
```shell
python little_nurse/finetune.py \
    --model_path "./Checkpoints/Llama-2-7b-chat-hf" \ # 基座模型llama2
    --data_path "./Data/datasets/aiwei.json" \ # 微调所使用的数据集
    --output_dir "./outputs/aiwei_lora_64" \
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

```
