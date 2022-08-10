import os
import sys
import argparse

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoConfig

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


class MyBert(nn.Module):

    def __init__(self, args) -> None:
        super(MyBert, self).__init__()
        pass

    def forward(self, x):
        pass


def get_bert(args):
    """_summary_

    Args:
        args.pretrained_model (_type_): 预训练模型的类型， BERT or SBERT
        args.bert (_type_): 与训练模型的名字
    """
    if args.pretrained_model == 'SBERT':
        sbert = SentenceTransformer(SBERT_CLASS[args.bert])
    elif args.pretrained_model == 'BERT' or args.pretrained_model == 'PairSupCon':
        config = AutoConfig.from_pretrained(BERT_CLASS[args.bert])
        model = AutoModel.from_pretrained(BERT_CLASS[args.bert], config=config)
    else:
        raise NotImplementedError
    
    if args.use_noise:
        for name, para in model.bert.named_parameters():
            model.bert.state_dict()[name][:] += (torch.rand(para.size()) - 0.5)*args.noise_lambda*torch.std(para)

    return model
