import os
import ssl
import requests
from huggingface_hub import configure_http_backend
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import urllib3


urllib3.disable_warnings()

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

# ✅ Disable SSL verification globally (use with caution!)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

# Function to download a Hugging Face model
def download_model(model_name):
    print(f"Downloading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True,
                                              )
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    print(f"Model and tokenizer for '{model_name}' downloaded successfully.")
    return tokenizer, model

# Function to download a Hugging Face dataset
def download_dataset(dataset_name, split='train'):
    print(f"Downloading dataset: {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    print(f"Dataset '{dataset_name}' downloaded successfully.")
    return dataset

# Example usage
if __name__ == "__main__":
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    dataset_name = "yahma/alpaca-cleaned"

    tokenizer, model = download_model(model_name)
    dataset = download_dataset(dataset_name)

    print("\nSample from dataset:")
    print(dataset[0])