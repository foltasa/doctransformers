from datasets import (
    Dataset,
    DatasetDict,
    )
import polars as pl
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from . import (
    DocDataset,
)
from typing import Optional

def create_doc_dataset(
    data : Dataset|DatasetDict,
    tokenizer : PreTrainedTokenizerFast,
    end_of_sentence_tokens : list[str] = [".", "!", "?"],
    max_tokens : Optional[int] = None,
):
    """Creates a DocDataset

    Args:
        data (Dataset | DatasetDict): The formated original dataset
        tokenizer (PreTrainedTokenizerFast): A huggingface tokenizer for chunking. 
        end_of_sentence_tokens (list[str], optional): Documents are only splitted at the end of an sentence. Defaults to [".", "!", "?"].
        max_tokens (Optional[int], optional): How many a chunk at most. If None the max token value of the tokenizer is used. Defaults to None.

    Returns:
        [`DocDataset`]
    """
    eos_ids = []
    for t in end_of_sentence_tokens:
        eos_ids.append(tokenizer.convert_tokens_to_ids(t)) 
    eos_ids = eos_ids
    
    def _tokenize(examples):
        encoding = tokenizer(
            examples["text"], 
            padding=False, 
            add_special_tokens=False, 
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            verbose=False,
            )
        return encoding
    
    data = _create_doc_id(data)
    encodings = data.map(
        _tokenize, batched=True, remove_columns="text",
    )

    chunks = pl.DataFrame(schema={"doc_id" : str, "chunk" : int, "label" : pl.Int64, "text" : str})
    if isinstance(encodings, DatasetDict):
        for split in encodings:
            chunks = _create_chunks(
                tokenizer=tokenizer,
                eos_ids=eos_ids,
                encodings=encodings[split],
                chunks=chunks,         
                max_tokens=max_tokens,   
                )
    elif isinstance(encodings, Dataset):
        chunks = _create_chunks(
                tokenizer=tokenizer,
                eos_ids=eos_ids,
                encodings=encodings,
                chunks=chunks,  
                max_tokens=max_tokens,          
                )
    else:
        raise ValueError(f"Data should be a Dataset or a DatasetDict. Is of type {type(encodings)}.")
    
    docdata = DocDataset()
    docdata["docs"] = data
    docdata["chunks"] = Dataset.from_dict(chunks.to_dict())
    return  docdata
   
def _create_chunks(
    tokenizer : PreTrainedTokenizerFast,
    eos_ids : list[int],
    encodings : Dataset,
    chunks : pl.DataFrame,
    max_tokens : Optional[int] = None,
    ):    
    
    if max_tokens is None:
        max_tokens = tokenizer.model_max_length - 2  
    else:
        max_tokens = max_tokens - 2
        
    def _search_end_of_sentence(input_ids : list[int]):
        indexes = [i for i, x in enumerate(input_ids[:max_tokens]) if x in eos_ids]
        if not len(indexes):
            index = max_tokens
        else:
            index = max(indexes) + 1
        return input_ids[:index], input_ids[index:]
            
    for row in tqdm(encodings, desc="Create Chunks"):
        doc_id = row["doc_id"]
        chunk = 0
        ids = row["input_ids"]
        label = row["label"]
        while True:
            if len(ids) <= max_tokens:
                chunks = chunks.vstack(
                        pl.DataFrame({
                            "doc_id" : doc_id,
                            "chunk" : chunk,
                            "label" : label,
                            "text" : tokenizer.decode(ids),
                        })
                    )
                break
            else:
                chunk_ids, ids = _search_end_of_sentence(input_ids=ids)
                chunks = chunks.vstack(
                        pl.DataFrame({
                            "doc_id" : doc_id,
                            "chunk" : chunk,
                            "label" : label,
                            "text" : tokenizer.decode(chunk_ids)
                        })
                    )
                chunk += 1
    return chunks

def _create_doc_id(data : Dataset|DatasetDict):
    if isinstance(data, DatasetDict):
        for split in data:
            if "doc_id" in data[split].column_names:
                pass
            else:
                data[split] = data[split].add_column("doc_id", column=[f"doc_{split}_{x}" for x in range(data[split].num_rows)])
    elif isinstance(data, Dataset):
        if "doc_id" in data[split].column_names:
            pass
        else:
            data = data.add_column("doc_id", column=[f"doc_{x}" for x in range(data.num_rows)])
    return data
    
