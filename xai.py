from dotenv import load_dotenv
import os

from datasets import load_dataset
from transformers import ResNetForImageClassification

# model = ResNetForImageClassification.from_pretrained()

load_dotenv()
token = os.getenv("token")
dataset = load_dataset("ILSVRC/imagenet-1k", token=token, cache_dir="./venv/XAI_Dataset")
