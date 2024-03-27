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


# model_name = "yahma/llama-7b-hf"
# model_name = "luodian/llama-13b-hf"
# model_name = "alexl83/LLaMA-33B-HF"
# model_name = "Enoch/llama-65b-hf"
model_name = "joyfine/vicuna-7b-fine-tuning_truthfulQA_512_20"

tokenizer = llama.LLaMATokenizer.from_pretrained(model_name)
model = llama.LLaMAForCausalLM.from_pretrained(model_name, force_download=True, resume_download=False)