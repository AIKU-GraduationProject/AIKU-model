from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="bert-base-multilingual-cased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default="/opt/ml/input/data/train_dataset", metadata={"help": "The name of the dataset to use."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    train_retrieval: bool = field(
        default=False,
        metadata={"help": "Whether to train sparse/dense embedding (prepare for retrieval)."},
    )
    eval_retrieval: bool = field(
        default=False,
        metadata={"help": "Whether to run passage retrieval using sparse/dense embedding )."},
    )
    embedding_mode: str = field(
        default='bm25_new',
    )
    topk: int = field(
        default=1,
    )
    include_korquad: bool = field(
        default=False,
    )
