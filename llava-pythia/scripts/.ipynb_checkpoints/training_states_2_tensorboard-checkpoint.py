import torch
from torch.utils.tensorboard import SummaryWriter
import os
import json

def main():
    # 创建SummaryWriter对象
    pythia = "410M"
    log_p = f'/data/private/wenjj/llava-pythia/checkpoint_all/pythia_{pythia}/vanilla_pythia_pt_f_vit/llavaPythia-v0-robot-action-w_state_huber/log'
    
    trainint_state_p = f"/data/private/wenjj/llava-pythia/checkpoint_all/pythia_{pythia}/vanilla_pythia_pt_f_vit/llavaPythia-v0-robot-action-w_state_huber/trainer_state.json"
    
    os.makedirs(log_p, exist_ok=True)

    writer = SummaryWriter(log_dir=log_p)
    
    # 假设loss_data是你保存的loss数据，格式为字典
    with open(trainint_state_p, "r") as f:
        data = json.load(f)

    # 将loss数据写入SummaryWriter
    for each in data['log_history']:
        if not 'loss' in each.keys():
            continue
        step, loss = each['step'], each['loss']
        writer.add_scalar('train/loss', loss, step)

    # 关闭SummaryWriter对象
    writer.close()

if __name__ == "__main__":
    main()
