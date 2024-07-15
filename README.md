# Doctransformers (beta)

This library is based on the [Transformers](https://github.com/huggingface/transformers) library by HuggingFace and provides tools for training models on longer documents (multiple pages) or books. `doctransformers` allows you to disassemble documents into small chunks, train a model to embed each chunk, and pool them to generate document vectors ready for downstream tasks. This procedure improves model performances of medium-length texts by avoiding truncation and enables the usage of documents/books that exceed most models' max token limits by far.

**Currently in beta and restricted to the [BertForSequenceClassification](https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertForSequenceClassification) model architecture**

## Setup 
TODO

## Usage

1. Load a [Dataset](https://huggingface.co/docs/datasets/index) and perform all required data operations as train test splits.
2. Load a [PreTrainedTokenizerFast](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) and a huggingface model for the specific downstream task (currently only [BertForSequenceClassification](https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertForSequenceClassification) supported)
3. Create a `DocDataset` using the `create_doc_dataset` function
4. Create and train the `DocTrainer` 

# Example

This example showcases how to use `doctransformers` using the [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) dataset. While this library is primarily designed for extended reports that exceed BERT's 512 max tokens by far, medium-length IMDb reviews allow performance comparisons with approaches relying on truncation.

## Create a DocDataset

```python

from datasets import load_dataset
from transformers import AutoTokenizer
from doctransformers import create_doc_dataset

# Loading the data
data = load_dataset("stanfordnlp/imdb")
data.pop("unsupervised") # Drop unnecessary data to reduce computation time
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

docdata = create_doc_dataset(data=data, tokenizer=tokenizer, max_tokens=128) # Splitting docs into chunks currently takes some time; working on optimizations
docdata.save_to_disk("example/data") # Save data locally

```

## Train a model

In this example I use the same base model and hyperparameters as the popular [textattack/bert-base-uncased-imdb](https://huggingface.co/textattack/bert-base-uncased-imdb) model with two exceptions:
- I use 128 token chunks instead of truncating the reviews after 128 tokens
- The training epochs are reduced to 1 (from 5)

```python

from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments
from doctransformers import DocDataset, DocTrainer
import evaluate
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess the docdataset
docdata = DocDataset.load_from_disk("example/data")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
docdata.preprocess(tokenizer=tokenizer)

# Prepare TrainingArguments as you would for a transformers Trainer
acc = evaluate.load("accuracy")
id2label = {1: "POS", 0: "NEG"}
label2id = {"POS": 1, "NEG": 0}

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                num_labels=2, id2label=id2label, label2id=label2id).to("cuda")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return acc.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="example/model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    num_train_epochs=1,
    weight_decay=0.01,
)

# Init doctrainer
clf = RandomForestClassifier(n_jobs=8, verbose=1) # The random forest classifier to classify the documents 

trainer = DocTrainer(
    model=model,
    doc_classifier=clf,
    data_collator=data_collator,
    args=training_args,
    tokenizer=tokenizer,
    doc_dataset=docdata,
    compute_metrics=compute_metrics,
)

# Train the BERT model to embedd chunks
trainer.train()

# Train document classifier
trainer.train_head() # Accuracy 0.9502

```

The accuracy of 0.9502 exceeds the 0.8909 of [textattack/bert-base-uncased-imdb](https://huggingface.co/textattack/bert-base-uncased-imdb), demonstrating improved performance without truncation. While the performance gain is relatively modest for shorter IMDb reviews, it greatly increases when using documents consisting of multiple pages. 

# TODO
- Performance optimizations
- Building a pipeline for easy use of trained models
- Allow the usage of other models besides BERT
- Extend the package for other tasks besides classification 




