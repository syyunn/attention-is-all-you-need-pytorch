import time
import random

import numpy as np
import torch

# Load model
from transformer.Infersent import InferSent

# Download nltk: punkt
import nltk


def infersent():
    nltk.download('punkt')

    model_version = 2
    MODEL_PATH = "ptds/infersent%s.pkl" % model_version
    params_model = {'bsize': 64,
                    'word_emb_dim': 300,
                    'enc_lstm_dim': 2048,
                    'pool_type': 'max',
                    'dpout_model': 0.0,
                    'version': model_version}

    model = InferSent(params_model)

    model.load_state_dict(torch.load(MODEL_PATH))

    # Keep it on CPU or put it on GPU
    use_cuda = True
    model = model.cuda() if use_cuda else model

    # If infersent1 -> use GloVe embeddings.
    # If infersent2 -> use InferSent embeddings.
    W2V_PATH = '../dataset/GloVe/glove.840B.300d.txt' if model_version == 1 \
        else '/home/zachary/projects/InferSent/dataset/fastText/crawl-300d-2M-subword.vec'
    model.set_w2v_path(W2V_PATH)

    # Load embeddings of K most frequent words
    model.build_vocab_k_words(K=100000)

    # Freeze the parameter
    for param in model.parameters():
        param.requires_grad = False
    print("freezing!")

    return model

