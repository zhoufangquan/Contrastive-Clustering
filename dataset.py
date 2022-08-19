import os
import pandas as pd
import torch
import torch.utils.data as util_data
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoConfig

import argparse
from utils import yaml_config_hook
from modules import bert, contrastive_loss, network

BERT_CLASS = {
    "bertbase": 'bert-base-uncased',
    "bertlarge": 'bert-large-uncased',
    "pairsupcon-base": "aws-ai/pairsupcon-bert-base-uncased",
    "pairsupcon-large": "aws-ai/pairsupcon-bert-large-uncased",
}

SBERT_CLASS = {
    "distilbert": 'distilbert-base-nli-mean-tokens',
    "bertbase": 'bert-base-nli-mean-tokens',
    "bertlarge": 'bert-large-nli-mean-tokens',
}


class MyDataset(Dataset):
    def __init__(self, args) -> None:
        data_path = os.path.join(args.data_dir, args.data_name+'.csv')
        df_data = pd.read_csv(data_path)
        text = df_data['text'].fillna('.').tolist()
        text1 = df_data['text1'].fillna('.').tolist()
        text2 = df_data['text2'].fillna('.').tolist()
        self.label = df_data['label'].astype(int).tolist()  # 数据标签从零开始
        # self.label = df_data['label'].astype(int).tolist() - 1  # 数据标签从零开始
    
        # evaluate vanilla BERT or PairSupCon
        if args.pretrained_model in ["BERT", "PairSupCon"]:
            tokenizer = AutoTokenizer.from_pretrained(BERT_CLASS[args.bert])
        elif args.pretrained_model == "SBERT":  # evaluate SentenceBert
            raise NotImplementedError
        else:
            raise Exception(
                "please specify the pretrained model you want to evaluate")

        tokenized_text = tokenizer(
            text,
            padding="max_length",
            max_length=args.max_len,
            truncation=True
        )
        df_tmp = pd.DataFrame.from_dict(tokenized_text, orient="index").T
        self.tokenized_text = df_tmp.to_dict(orient="records")

        tokenized_text1 = tokenizer(
            text1,
            padding="max_length",
            max_length=args.max_len,
            truncation=True
        )
        df_tmp = pd.DataFrame.from_dict(tokenized_text1, orient="index").T
        self.tokenized_text1 = df_tmp.to_dict(orient="records")

        tokenized_text2 = tokenizer(
            text2,
            padding="max_length",
            max_length=args.max_len,
            truncation=True
        )
        df_tmp = pd.DataFrame.from_dict(tokenized_text2, orient="index").T
        self.tokenized_text2 = df_tmp.to_dict(orient="records")


    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return {'text': self.tokenized_text[index],
                'text1': self.tokenized_text1[index],
                'text2': self.tokenized_text2[index],
                'label': self.label[index]}

def collate_fn(batch):
    max_len = 0
    for X in batch:
        max_len = max(
            max_len,
            sum(X['text']['attention_mask']),
            sum(X['text1']['attention_mask']),
            sum(X['text2']['attention_mask']),
        )
        
    texts = dict()
    for name in ['text', 'text1', 'text2']:
        all_input_ids = torch.tensor([x[name]['input_ids'][:max_len] for x in batch])
        all_token_type_ids = torch.tensor([x[name]['token_type_ids'][:max_len] for x in batch])
        all_attention_mask = torch.tensor([x[name]['attention_mask'][:max_len] for x in batch])
        texts[name] = {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask,
            'token_type_ids': all_token_type_ids
        }
    
    all_labels = torch.tensor([x['label'] for x in batch])

    return (
        texts['text'],  # 原始数据
        texts['text1'], # 数据增强 1
        texts['text2'], # 数据增强 2
        all_labels  # 数据的类别标签
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MyDataset(args)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    bert = bert.get_bert(args)
    model = network.Network(args, bert)
    model = model.to(device)

    
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, device).to(device)
    criterion_cluster = contrastive_loss.ClusterLoss(args.class_num, args.cluster_temperature, device).to(device)

    for i, (x, x_i, x_j, label) in enumerate(data_loader):
        for x in x_i.keys():
            x_i[x] = x_i[x].to(device)
            x_j[x] = x_j[x].to(device)
        print(x_i, x_j)
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster
        print()
        break
