#!/usr/bin/env python

import itertools
import math
import numpy as np
import os
import pickle
import random
import re
import string
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
from transformers import (
    AdamW,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)
from bertviz import model_view
seed = 211
np.random.seed(seed)
torch.manual_seed(seed)


def clean_email(email):
    # Remove URLs
    email = re.sub(r'http\S+|www.\S+', '', email)

    # Remove special characters but keep letters, numbers, and spaces
    email = re.sub(r'[^a-zA-Z0-9\s]', '', email)

    # Remove words longer than 20 characters
    email = ' '.join(word for word in email.split() if len(word) <= 20)

    return email
    
def example_usage_clean():
    print("Hello, welcome to the Bert-Based E-mail Classifier! This is an example usage for the clean_email method. We'll clean this email:\n")
    email = "Hello, this is a sample email! Please visit https://example.com for more info. Thanks :)"
    print(email)
    print("\nCleaned email:\n")
    cleaned_email = clean_email(email)
    print(cleaned_email)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def check_model_dir():
    out_dir = '../model'
    if not os.path.exists(out_dir):
        print("path to model does not exist, please reclone repository")
        exit(1)
    
def get_top_tokens_from_text(text, tokenizer, model, device, top_k=5):
    text = clean_email(text)
    inputs = tokenizer.encode(text, return_tensors='pt').to(device)

    outputs = model(inputs)
    attentions = outputs[-1]

    # srack attention across heads
    attn_weights = torch.stack(attentions).squeeze()

    #attn_weights.shape = [6,12,len,len]
    # Sum attention weights across heads and layers
    attn_weights_sum = attn_weights.sum(dim=1).sum(dim=0)  # Shape: [seq_len, seq_len]

    #attn_weights_sum.shape = [len,len]

    cls_attn_weights = attn_weights_sum[0]  # Attention weights of the [CLS] token

    # Identify top k important tokens, excluding special tokens and punctuation
    top_tokens = []
    top_indices = []
    for value, index in zip(*torch.topk(cls_attn_weights, len(cls_attn_weights))):
        token = tokenizer.convert_ids_to_tokens(inputs[0][index].item())
        if token not in tokenizer.all_special_tokens and token not in string.punctuation:
            top_tokens.append(token)
            top_indices.append(index)
        if len(top_tokens) == top_k:  # Stop when you have 5 non-special, non-punctuation tokens
            break
    highlighted_text = text
    for token, idx in zip(top_tokens, top_indices):
        word = tokenizer.convert_ids_to_tokens(inputs[0][idx].item()).strip("##")
        highlighted_text = highlighted_text.replace(word, f"[{word}]")

    return top_tokens, highlighted_text

def classify_email(text, do_cleaning=True):
  if(do_cleaning):
    text = clean_email(text)
  max_length = 300
  tokenizer_name = 'distilbert-base-uncased'
  model_path = 'model/'
  tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
  model = DistilBertForSequenceClassification.from_pretrained(model_path)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  inputs = tokenizer.encode(text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt').to(device)
  model.eval()
  with torch.no_grad():
    outputs = model(inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

  return "Legitimate" if prediction == 1 else "Phishing"