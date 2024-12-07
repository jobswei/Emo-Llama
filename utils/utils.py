
import torch.distributed as dist
import json

def load_json(path:str):
    return json.load(open(path,"r"))
def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)