import json
import os
import os.path as osp
import tqdm
EmoLLM_Data = []
root = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/PaddleNLP/Data/datasets"
out_root = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/PaddleNLP/Data/new_formated_data"
files = [file for file in os.listdir(root) if file.endswith(".json")]
# files = ["aiwei.json"]
for file in tqdm.tqdm(files):
    with open(osp.join(root,file), 'r') as f:  
        data = json.load(f)
    new_data = []
    for line in data:
        if "conversation" in line: 
            conversations = line["conversation"]
            if not conversations:
                continue
            assert "input" in conversations[0] and "output" in conversations[0], file
            if "system" not in conversations[0]:
                conversations[0]["system"] = ""
            new_data.append({"conversation":conversations})
        elif "instruction" in line and "input" in line:
            new_data.append({"conversation":[{"system":"","input":line["instruction"]+line["input"],"output":line["output"]}]})
        elif "prompt" in line:
            new_data.append({"conversation":[{"system":"","input":line["prompt"],"output":line["completion"]}]})
    with open(osp.join(out_root,file), 'w', encoding='utf-8') as f:
        json.dump(new_data,f,ensure_ascii=False)