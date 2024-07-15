
import re
import os
import warnings
import fsspec
import json
import posixpath

from typing import Dict, Literal, Optional, Union, List
from datasets import DatasetDict, Dataset
from datasets.filesystems import extract_path_from_uri, is_remote_filesystem
from datasets.utils.typing import PathLike
from pathlib import Path
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import torch
import polars as pl

DOCDATASET_INFO_FILENAME = "doc_dataset.json"

class DocDataset(DatasetDict):
    
    """A Dataset designed for large documents. """
    
    def train_test_split(self, **kwargs):
        """Splits the documents into training and test sample. 
        """
        self["docs"] = self["docs"].train_test_split(**kwargs)

    def __repr__(self):
        repr = "\n".join([f"{k}: {v}" for k, v in self.items()])
        repr = re.sub(r"^", " " * 4, repr, 0, re.M)
        return f"DocDataset({{\n{repr}\n}})"
    
    def tokenize_chunks(self, tokenizer : PreTrainedTokenizerBase, **kwargs):
        """Tokenizes the chunks
        Args:
            tokenizer (PreTrainedTokenizerBase): A transformers tokenizer.
        """
        def _tokenize(row):
            return tokenizer(row["text"], **kwargs)
        self["chunks"] = self["chunks"].map(_tokenize)
    
    def embedd(self, 
               model : PreTrainedModel,
               tokenizer : PreTrainedTokenizerBase,
               redo_embeddings : bool = False,
               ):
        """Embedds the chunks and documents using a fine-tuned model for document classification.

        Args:
            model (PreTrainedModel): A fine-tuned model
            tokenizer (PreTrainedTokenizerBase): A huggingface tokenizer
            redo_embeddings (bool, optional): Delete already existing embeddings? Defaults to False.
        """
        if redo_embeddings or not "embeddings" in self["chunks"].column_names:
            self["chunks"] = self._embedd_chunks(model=model, tokenizer=tokenizer)
        else:
            print("Chunks already embedded.")
        if isinstance(self["docs"], DatasetDict):
            for name in self["docs"]:
                if redo_embeddings or not "embeddings" in self["docs"][name].column_names:
                    self["docs"][name] = self._embedd_docs(self["docs"][name])
                else:
                    print(f"Docs: {name} already embedded.")
        else:
            if redo_embeddings or not "embeddings" in self["docs"].column_names:
                self["docs"] = self._embedd_docs(self["docs"])
            else:
                print("Docs already embedded.")
            
    def _embedd_docs(self, docs : Dataset):
        chunks = pl.from_pandas(self["chunks"].to_pandas())
        def _embedd_docs(row) -> Dict[str, torch.Tensor]:
            doc_id = row["doc_id"]
            data = chunks.filter(pl.col("doc_id").eq(doc_id))
            tensor = data.select("embeddings").with_columns(
                pl.col("embeddings").list.to_struct()
            ).unnest("embeddings").mean().transpose()["column_0"].to_list()
            return {"embeddings" : tensor}
        return docs.map(_embedd_docs, desc="Embedding docs.")

    def _embedd_chunks(self, 
                       model : PreTrainedModel, 
                       tokenizer : PreTrainedTokenizerBase
                       ):
        model.base_model.eval()
        def _embedd_data(row):
            with torch.no_grad():
                output = model.base_model(**tokenizer(row["text"],
                                    return_tensors="pt", 
                                    padding=True,).to(model.device))
            return {"embeddings" : output["pooler_output"][0] }
        return self["chunks"].map(_embedd_data, desc="Embedding chunks.")     
    
    def get_embeddings_for_downstream_tasks(self, 
                        format : Literal["docs", "chunks"] = "chunks",
                        split : Optional[str] = None,
                        ):   
        if split is None:
            output = pl.from_pandas(self[format].to_pandas())
        else:
            output = pl.from_pandas(self[format][split].to_pandas())
        return output.select("embeddings"). \
            with_columns(pl.col("embeddings").list.to_struct()
                ).unnest("embeddings")
            
    @staticmethod
    def load_from_disk(
        dataset_dict_path: PathLike,
        fs="deprecated",
        keep_in_memory: Optional[bool] = None,
        storage_options: Optional[dict] = None,
    ):
        """
        Load a docdataset that was previously saved using [`save_to_disk`] from a filesystem using `fsspec.spec.AbstractFileSystem`.

        Args:
            dataset_dict_path (`str`):
                Path (e.g. `"dataset/train"`) or remote URI (e.g. `"s3//my-bucket/dataset/train"`)
                of the dataset dict directory where the dataset dict will be loaded from.
            fs (`fsspec.spec.AbstractFileSystem`, *optional*):
                Instance of the remote filesystem where the dataset will be saved to.

                <Deprecated version="2.8.0">

                `fs` was deprecated in version 2.8.0 and will be removed in 3.0.0.
                Please use `storage_options` instead, e.g. `storage_options=fs.storage_options`

                </Deprecated>

            keep_in_memory (`bool`, defaults to `None`):
                Whether to copy the dataset in-memory. If `None`, the
                dataset will not be copied in-memory unless explicitly enabled by setting
                `datasets.config.IN_MEMORY_MAX_SIZE` to nonzero. See more details in the
                [improve performance](../cache#improve-performance) section.
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.8.0"/>

        Returns:
            [`DocDataset`]
        """
        if fs != "deprecated":
            warnings.warn(
                "'fs' was deprecated in favor of 'storage_options' in version 2.8.0 and will be removed in 3.0.0.\n"
                "You can remove this warning by passing 'storage_options=fs.storage_options' instead.",
                FutureWarning,
            )
            storage_options = fs.storage_options

        fs_token_paths = fsspec.get_fs_token_paths(dataset_dict_path, storage_options=storage_options)
        fs: fsspec.AbstractFileSystem = fs_token_paths[0]

        if is_remote_filesystem(fs):
            dest_dataset_dict_path = extract_path_from_uri(dataset_dict_path)
            path_join = posixpath.join
        else:
            fs = fsspec.filesystem("file")
            dest_dataset_dict_path = dataset_dict_path
            path_join = os.path.join
            
        doc_dataset_info_path = path_join(dest_dataset_dict_path, DOCDATASET_INFO_FILENAME)
        
        if not fs.isfile(doc_dataset_info_path):
            raise FileNotFoundError(
                f"No such file: '{doc_dataset_info_path}'. Expected to load a `DocDataset` object, but provided path is not a `DocDataset`."
            )
            
        with fs.open(doc_dataset_info_path, "r", encoding="utf-8") as f:
            splits = json.load(f)["splits"]
            
        doc_dataset = DocDataset()
        for k in splits:
            doc_dataset_split_path = (
                dataset_dict_path.split("://")[0] + "://" + path_join(dest_dataset_dict_path, k)
                if is_remote_filesystem(fs)
                else path_join(dest_dataset_dict_path, k)
            )
            try:
                doc_dataset[k] = Dataset.load_from_disk(
                    doc_dataset_split_path, keep_in_memory=keep_in_memory, storage_options=storage_options
                )
            except FileNotFoundError:
                try:
                    doc_dataset[k] = DatasetDict.load_from_disk(
                        doc_dataset_split_path, keep_in_memory=keep_in_memory, storage_options=storage_options
                    )
                except FileNotFoundError:
                    raise FileNotFoundError(f"Expected to load `DatasetDict` or `Dataset` object in split '{k}'.")
        return doc_dataset
    
    def preprocess(self, tokenizer : PreTrainedTokenizerBase):
        """Preprocesses the text data for usage in Doctrainer.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer for preprocessing.
        """
        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True, padding=True)
        
        self["docs"] = self["docs"].map(preprocess_function, batched=True)
        self["chunks"] = self["chunks"].map(preprocess_function, batched=True)

    def save_to_disk(
            self,
            dataset_dict_path: PathLike,
            fs="deprecated",
            max_shard_size: Optional[Union[str, int]] = None,
            num_shards: Optional[Dict[str, int]] = None,
            num_proc: Optional[int] = None,
            storage_options: Optional[dict] = None,
        ):
            """
            Saves a dataset dict to a filesystem using `fsspec.spec.AbstractFileSystem`.

            Args:
                dataset_dict_path (`str`):
                    Path (e.g. `dataset/train`) or remote URI
                    (e.g. `s3://my-bucket/dataset/train`) of the dataset dict directory where the dataset dict will be
                    saved to.
                fs (`fsspec.spec.AbstractFileSystem`, *optional*):
                    Instance of the remote filesystem where the dataset will be saved to.

                    <Deprecated version="2.8.0">

                    `fs` was deprecated in version 2.8.0 and will be removed in 3.0.0.
                    Please use `storage_options` instead, e.g. `storage_options=fs.storage_options`

                    </Deprecated>

                max_shard_size (`int` or `str`, *optional*, defaults to `"500MB"`):
                    The maximum size of the dataset shards to be uploaded to the hub. If expressed as a string, needs to be digits followed by a unit
                    (like `"50MB"`).
                num_shards (`Dict[str, int]`, *optional*):
                    Number of shards to write. By default the number of shards depends on `max_shard_size` and `num_proc`.
                    You need to provide the number of shards for each dataset in the dataset dictionary.
                    Use a dictionary to define a different num_shards for each split.

                    <Added version="2.8.0"/>
                num_proc (`int`, *optional*, default `None`):
                    Number of processes when downloading and generating the dataset locally.
                    Multiprocessing is disabled by default.

                    <Added version="2.8.0"/>
                storage_options (`dict`, *optional*):
                    Key/value pairs to be passed on to the file-system backend, if any.

                    <Added version="2.8.0"/>
            """
            
            if fs != "deprecated":
                warnings.warn(
                    "'fs' was deprecated in favor of 'storage_options' in version 2.8.0 and will be removed in 3.0.0.\n"
                    "You can remove this warning by passing 'storage_options=fs.storage_options' instead.",
                    FutureWarning,
                )
                storage_options = fs.storage_options

            fs_token_paths = fsspec.get_fs_token_paths(dataset_dict_path, storage_options=storage_options)
            fs: fsspec.AbstractFileSystem = fs_token_paths[0]
            is_local = not is_remote_filesystem(fs)
            path_join = os.path.join if is_local else posixpath.join

            if num_shards is None:
                num_shards = {k: None for k in self}
            elif not isinstance(num_shards, dict):
                raise ValueError(
                    "Please provide one `num_shards` per dataset in the dataset dictionary, e.g. {{'train': 128, 'test': 4}}"
                )

            if is_local:
                Path(dataset_dict_path).resolve().mkdir(parents=True, exist_ok=True)
            else:
                fs.makedirs(dataset_dict_path, exist_ok=True)

            with fs.open(path_join(dataset_dict_path, DOCDATASET_INFO_FILENAME), "w", encoding="utf-8") as f:
                json.dump({"splits": list(self)}, f)
            for k, dataset in self.items():
                dataset.save_to_disk(
                    path_join(dataset_dict_path, k),
                    num_shards=num_shards.get(k),
                    max_shard_size=max_shard_size,
                    num_proc=num_proc,
                    storage_options=storage_options,
                )

    @property
    def column_names(self) -> Dict[str, List[str]]:
        """Names of the columns in the dataset.
        """
        return {k: dataset.column_names for k, dataset in self.items()}
    
    def rename_column(self, original_column_name: str, new_column_name: str):
        """
        Rename a column in the dataset, and move the features associated to the original column under the new column
        name.

        Args:
            original_column_name (`str`):
                Name of the column to rename.
            new_column_name (`str`):
                New name for the column.
            new_fingerprint (`str`, *optional*):
                The new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments.

        Returns:
            [`DocDataset`]: A copy of the dataset with a renamed column. """
            
        self["docs"] = self["docs"].rename_column(original_column_name=original_column_name, new_column_name=new_column_name)
        self["chunks"] = self["chunks"].rename_column(original_column_name=original_column_name, new_column_name=new_column_name)

    