import torch
from pprint import pprint
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
from einops import rearrange
import numpy as np
import pickle
import os
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset
import pickle as pkl
import torch.nn.functional as F

import sys
sys.path.append('../')
from utils import *


model_name = "yahma/llama-7b-hf"
url = "https://huggingface.co/api/datasets/truthful_qa/parquet/multiple_choice/validation/0.parquet"
dataset = load_dataset('parquet', data_files=url)['train']

tokenizer = llama.LLaMATokenizer.from_pretrained(model_name)
model = llama.LLaMAForCausalLM.from_pretrained(model_name, force_download=True, resume_download=False)