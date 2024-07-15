from typing import Callable, Dict, List, Optional, Tuple, Union
from pick import Option
import torch
from torch import nn
from transformers import (
    Trainer, PreTrainedModel, PreTrainedTokenizerFast, TrainingArguments,
    DataCollator, TrainerCallback, EvalPrediction,
)
from datasets import DatasetDict
import polars as pl
from ..datasets import DocDataset
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

class DocTrainer(Trainer):
    
    """
    DocTrainer is based on the 🤗 Transformers and uses Docdatasets to train 🤗 models on large documents or books.
    
    Args:
        model ([`PreTrainedModel`] or `torch.nn.Module`, *optional*):
            The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.

            <Tip>

            [`Trainer`] is optimized to work with the [`PreTrainedModel`] provided by the library. You can still use
            your own models defined as `torch.nn.Module` as long as they work the same way as the 🤗 Transformers
            models.
        doc_classifier ([`RandomForestClassifier`], *optional*):
            The model to classify embedded documents. If none provided a classifier with default settings will be 
            initialized.
        args ([`TrainingArguments`], *optional*):
            The arguments to tweak for training. Will default to a basic instance of [`TrainingArguments`] with the
            `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
        doc_dataset ([`DocDataset`]):
            The DocDataset.
        data_collator (`DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
            default to [`default_data_collator`] if no `tokenizer` is provided, an instance of
            [`DataCollatorWithPadding`] otherwise.
        train_dataset (`torch.utils.data.Dataset` or `torch.utils.data.IterableDataset`, *optional*):
            The dataset to use for training. If it is a [`~datasets.Dataset`], columns not accepted by the
            `model.forward()` method are automatically removed.

            Note that if it's a `torch.utils.data.IterableDataset` with some randomization and you are training in a
            distributed fashion, your iterable dataset should either use a internal attribute `generator` that is a
            `torch.Generator` for the randomization that must be identical on all processes (and the Trainer will
            manually set the seed of this `generator` at each epoch) or have a `set_epoch()` method that internally
            sets the seed of the RNGs used.
        eval_dataset (Union[`torch.utils.data.Dataset`, Dict[str, `torch.utils.data.Dataset`]), *optional*):
             The dataset to use for evaluation. If it is a [`~datasets.Dataset`], columns not accepted by the
             `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each
             dataset prepending the dictionary key to the metric name.
        tokenizer ([`PreTrainedTokenizerBase`], *optional*):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs to the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        model_init (`Callable[[], PreTrainedModel]`, *optional*):
            A function that instantiates the model to be used. If provided, each call to [`~Trainer.train`] will start
            from a new instance of the model as given by this function.

            The function may have zero argument, or a single one containing the optuna/Ray Tune/SigOpt trial object, to
            be able to choose different architectures according to hyper parameters (such as layer count, sizes of
            inner layers, dropout probabilities etc).
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values.
        callbacks (List of [`TrainerCallback`], *optional*):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](callback).

            If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*): A tuple
            containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your model
            and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.

    Important attributes:

        - **model** -- Always points to the core model. If using a transformers model, it will be a [`PreTrainedModel`]
          subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under `DeepSpeed`,
          the inner model is wrapped in `DeepSpeed` and then again in `torch.nn.DistributedDataParallel`. If the inner
          model hasn't been wrapped, then `self.model_wrapped` is the same as `self.model`.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
          to `False` if model parallel or deepspeed is used, or if the default
          `TrainingArguments.place_model_on_device` is overridden to return `False` .
        - **is_in_train** -- Whether or not a model is currently running `train` (e.g. when `evaluate` is called while
          in `train`)

    """

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        doc_classifier : Optional[RandomForestClassifier] = None,
        args: Optional[TrainingArguments] = None,
        doc_dataset : DocDataset = None,
        train_column : Optional[str] = "train",
        eval_column : Optional[str] = "test",
        data_collator: Optional[DataCollator] = None,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        
        
        """_summary_

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        
        self.train_column = train_column
        self.eval_column = eval_column
        columns = doc_dataset.column_names.keys()
        if not "docs" in columns:
            raise ValueError("DocTrainer needs a column 'docs' in the DocDataset.")
        if not "chunks" in columns:
            raise ValueError("DocTrainer needs a column 'chunks' in the DocDataset.")
        if isinstance(doc_dataset["docs"], DatasetDict):
            self.train_docs = doc_dataset["docs"][train_column]
            if not eval_column is None:
                self.eval_docs = doc_dataset["docs"][eval_column]
            else:
                self.eval_docs = None
                
        else:
            self.train_docs = doc_dataset["docs"]
            self.eval_docs = None
        self.doc_dataset = doc_dataset
        if doc_classifier is None:
            self.doc_classifier = RandomForestClassifier()
        else:
            self.doc_classifier = doc_classifier

        train_dataset, eval_dataset = self._select_chunks()
        super().__init__(
            model = model,
            args = args,
            data_collator = data_collator,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            tokenizer = tokenizer,
            model_init = model_init,
            compute_metrics = compute_metrics,
            callbacks = callbacks,
            optimizers = optimizers,
            preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        )
    
    def train_head(self):
        if not self._is_embedded():
            self.doc_dataset.embedd(model=self.model, tokenizer=self.tokenizer)
        print("Fitting classifier")
        self.doc_classifier.fit(
            self.doc_dataset.get_embeddings_for_downstream_tasks(format="docs", split=self.train_column), 
            self.doc_dataset["docs"][self.train_column]["label"],
            )
        print("Testing on eval data")
        if not self.eval_dataset is None:
            preds = self.doc_classifier.predict(
                self.doc_dataset.get_embeddings_for_downstream_tasks(format="docs", split=self.eval_column)
            )
            labels = self.doc_dataset["docs"][self.eval_column]["label"]
            return pl.DataFrame({
                "mcc" : metrics.matthews_corrcoef(labels, preds),
                "accuracy" : metrics.accuracy_score(labels, preds),
                "f1-score" : metrics.f1_score(labels, preds),
        })
            
    def _is_embedded(self):
        if (not "embeddings" in self.doc_dataset["chunks"].column_names or
            not "embeddings" in self.doc_dataset["docs"][self.train_column].column_names or
            not "embeddings" in self.doc_dataset["docs"][self.eval_column].column_names):
            return False
        else:
            return True
            
    def _select_chunks(
        self,
        ):            
        doc_ids = self.train_docs["doc_id"]
        train_dataset = self.doc_dataset["chunks"].filter(lambda x : True if x["doc_id"] in doc_ids else False,
                                                          desc="Selecting chunks",
                                                          )
        if not self.eval_docs is None:
            doc_ids = self.eval_docs["doc_id"]
            eval_dataset = self.doc_dataset["chunks"].filter(lambda x : True if x["doc_id"] in doc_ids else False,
                                                            desc="Selecting chunks",)
        else:
            eval_dataset = None
        return train_dataset, eval_dataset
