import os
import numpy as np
import torch
import time
import argparse
from modules import network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data

from dataset import *
from utils.utils import init_logger, seed_everything, logger

def train(device):
    loss_epoch = 0
    for step, (x, x_i, x_j, _) in enumerate(data_loader):  # [text, text1, text2, label]
        optimizer.zero_grad()
        for x in x_i.keys():
            x_i[x] = x_i[x].to(device)
            x_j[x] = x_j[x].to(device)
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            # print(f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
            logger.info(f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
        loss_epoch += loss.item()
    return loss_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置保存断点的地方
    args.check_point_path = os.path.join(args.check_point_path, args.data_name)
    if not os.path.exists(args.check_point_path):
        os.makedirs(args.check_point_path)
    
    # 设置保存日志的地方
    args.log_dir = os.path.join(args.log_dir, args.data_name)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    init_logger(
        log_file=args.log_dir+"/{}.log".format(time.strftime("%Y-%m-%d %Hh%Mmin", time.localtime())))

    seed_everything(args.seed)

    # prepare data
    dataset = MyDataset(args)

    data_loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    
    # initialize model
    bert = bert.get_bert(args)
    model = network.Network(args, bert)
    model = model.to(device)
    
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.reload:
        model_fp = os.path.join(args.check_point_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, device).to(device)
    criterion_cluster = contrastive_loss.ClusterLoss(args.class_num, args.cluster_temperature, device).to(device)
    
    # train
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(device)
        if epoch % 10 == 0:
            save_model(args, model, optimizer, epoch+1)
        # print(f"Epoch [{epoch+1}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
        logger.info(f"Epoch [{epoch+1}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
    save_model(args, model, optimizer, args.epochs)
