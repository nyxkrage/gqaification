from typing import Annotated, cast
from datasets import load_dataset, Dataset, IterableDataset
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from argumentation import Argumentation, ArgumentationModel

class DatasetDownloadArgs(ArgumentationModel):
    dataset: Annotated[
        str,
        Argumentation(description="Path to dataset"),
    ]
    num_tokens: Annotated[
        int,
        Argumentation(
            description="Number of tokens to download (approximate)",
        ),
    ]
    output_path: Annotated[
        str,
        Argumentation(
            description="Path to output dataset",
        ),
    ]
    tokenizer: Annotated[
        str,
        Argumentation(
            description="Path to tokenizer",
        ),
    ]
    max_length: Annotated[
        int,
        Argumentation(
            description="Maximum length of sequence",
        ),
    ] = 2048
    text_column: Annotated[
        str,
        Argumentation(
            description="Name of text column",
        ),
    ] = "text"


def main(args: DatasetDownloadArgs):
    dataset = cast(IterableDataset, load_dataset(args.dataset, streaming=True, split="train"))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def encode(examples):
        results = cast(dict[str, torch.Tensor], tokenizer(
            examples[args.text_column], max_length=args.max_length, truncation=True, padding=True, return_tensors="pt", return_length=True 
        ))
        results["labels"] = results["input_ids"].clone()
        results["labels"][results["attention_mask"] == 0] = -100
        return results

    dataset = dataset.map(encode, batched=True, remove_columns=[col for col in dataset.column_names if col != args.text_column])
    if args.text_column != "text":
        dataset = dataset.rename_column(args.text_column, "text")
    tokens = 0
    rows = 0
    it = iter(dataset)
    bar = tqdm(total=args.num_tokens, unit="tokens")
    while tokens <= args.num_tokens:
        row = next(it)
        row_tokens = int(row["length"])
        bar.update(row_tokens)
        rows += 1
        tokens += row_tokens
    bar.close()
    print(rows)

    # take rows of dataset
    subset = Dataset.from_list(list(dataset.take(rows)), features=dataset.features)
    subset.save_to_disk(args.output_path)


if __name__ == "__main__":
    Argumentation.run(main)
