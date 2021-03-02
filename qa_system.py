"""
This is the main I/O stream for the QA pipeline

1. setup Question Answering module
2. setup IR module
3. setup QA module
"""

import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer,BertForSequenceClassification,AdamW,BertConfig,get_linear_schedule_with_warmup
from lxml import etree as ET
import spacy
import scispacy
import en_core_sci_lg
from bs4 import BeautifulSoup as bs

import question_understanding


if __name__ == "__main__":

    # This is for cpu support for non-NVIDEA cuda-capable machines.
    spacy.prefer_gpu()

    #initialize model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('model', cache_dir=None)

    # load in BioBERT
    nlp = en_core_sci_lg.load()

    n = 0
    while(True):
        question_understanding.ask_and_receive(n,device,tokenizer,model,nlp)
        n += 1

