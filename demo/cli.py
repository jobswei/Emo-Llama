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
import warnings
warnings.filterwarnings("ignore")

model = AutoModelForCausalLM.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/Checkpoints/Llama-2-7b-chat-hf",                    
                torch_dtype=(torch.bfloat16),
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=False,)
tokenizer = AutoTokenizer.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/Checkpoints/Llama-2-7b-chat-hf")
tokenizer.pad_token = "[PAD]"
tokenizer.model_max_length = 4080
lora_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/Emo-Llama/outputs/aiwei_lora_64"
non_lora_trainables = torch.load(osp.join(lora_path, "non_lora_trainables.bin"))
model = PeftModel.from_pretrained(model, lora_path)
info = model.load_state_dict(non_lora_trainables, strict=False)
model = model.merge_and_unload()
model = model.to("cuda")

print()
system = "现在你是一个拥有丰富心理学知识的温柔御姐艾薇医生，我有一些心理问题，请你用专业的知识和温柔的口吻帮我解决。"
history = torch.Tensor(1,0).cuda().to(torch.int32)
uid = input("你的id: ",)
print()
if not uid:
    uid = "mbpM3"
while True:
    question = str(input(uid+": "))
    print()
    conversations = [
        {"from":"human","value":question},
        {"from":"gpt","value":""}
    ]

    input_ids = preprocess_llama_2([conversations],tokenizer,system_message=system,only_query=True)
    system = ""
    input_ids = input_ids.cuda()
    input_ids = torch.cat((history,input_ids),dim=1)
    mask = torch.ones_like(input_ids).cuda()
    output_ids = model.generate(input_ids=input_ids,attention_mask=mask)
    answer_ids = output_ids[:,input_ids.shape[1]:]
    outputs = tokenizer.decode(answer_ids.squeeze(0),skip_special_tokens=True)
    print("Assisant: ",outputs)
    print()

    history = output_ids