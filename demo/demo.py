import os
import os.path as osp
import sys
sys.path.append("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/Emo-Llama")
from typing import *
import copy
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.utils import *
from peft import PeftModel
from finetune import *
model = AutoModelForCausalLM.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/Checkpoints/Llama-2-7b-chat-hf",                    
                torch_dtype=(torch.bfloat16),
                low_cpu_mem_usage=False,)
tokenizer = AutoTokenizer.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/Checkpoints/Llama-2-7b-chat-hf")
tokenizer.pad_token = "[PAD]"
tokenizer.model_max_length = 4080
lora_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/PaddleNLP/outputs/aiwei-12-5"
non_lora_trainables = torch.load(osp.join(lora_path, "non_lora_trainables.bin"))
model = PeftModel.from_pretrained(model, lora_path)
info = model.load_state_dict(non_lora_trainables, strict=False)
model = model.merge_and_unload()
model = model.to("cuda")

print()
system = "现在你是一个拥有丰富心理学知识的温柔御姐艾薇医生，我有一些心理问题，请你用专业的知识和温柔的口吻帮我解决。"

# history = 
conversations = [
    {"from":"human","value":"医生，我有一些学习上的问题"},
    {"from":"gpt","value":""}
]

msg = preprocess_llama_2([conversations],tokenizer,system_message=system)
input_ids = msg["input_ids"].cuda()
mask = torch.ones_like(input_ids).cuda()
output_ids = model.generate(input_ids=input_ids,attention_mask=mask)
outputs = tokenizer.decode(output_ids.squeeze(0),skip_special_tokens=True)
print(outputs)

history = input_ids + output_ids
conversations = {
    
}
msg = preprocess_llama_2([conversations],tokenizer,system_message="")
input_ids = msg.squeeze(0)
input_ids = history + input_ids
output_ids = model.generate(input_ids=input_ids.unsqueeze(0))
outputs = tokenizer.decode(output_ids,skip_special_tokens=True)
print(outputs)

history = history + input_ids + output_ids