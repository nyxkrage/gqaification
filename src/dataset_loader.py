from datasets import load_dataset, concatenate_datasets
import json
import os
from transformers import AutoTokenizer

def load_dataset_config(dataset_config):
    """
    Process a single dataset from Huggingface Hub and count its tokens
    
    Args:
        dataset_config: Tuple of (dataset_path, config_dict)
        
    Returns:
        int: Total number of tokens in the dataset
    """
    dataset_path, config = dataset_config
    print("Processing dataset:", dataset_path)

    # Extract configuration options
    split = config.get('split', 'train')
    num_samples = config.get('num_samples', None)
    dataset_config = config.get('config', None)
    data_files = config.get("data_files", None)
    revision = config.get('revision', None)
    field = config.get("field", "text")

    # Load dataset from Huggingface Hub
    dataset = load_dataset(
        dataset_path,
        name=dataset_config,
        revision=revision,
        split=split,
        data_files=data_files,
    )
    # filter out any rows where the field is none
    dataset = dataset.filter(lambda x: x[field] is not None)

    # If num_samples is specified, take a subset
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    def rename_text_field(example):
        if (field != "text"):
            example["text"] = example[field]
            del example[field]
        return example
    dataset = dataset.map(rename_text_field)

    return dataset

def load_datasets_from_json(file):
    with open(file, "r") as f:
        configs = json.load(f)

    datasets = [load_dataset_config(config) for config in configs["configs"].items()]

    dataset = concatenate_datasets(datasets)
    dataset = dataset.shuffle()

    return dataset

if __name__ == "__main__":
    load_datasets_from_json("dataset_configs_correction.json")