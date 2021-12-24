import logging
import os
import sys
from datasets import load_metric, load_from_disk, Sequence, Value, Features, Dataset, DatasetDict, load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AutoModel,
    BertConfig
)
from tqdm import tqdm
from utils import tokenize
from retrieval import SparseRetrieval
import pandas as pd

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(f"data is from {data_args.dataset_name}")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    if data_args.include_korquad:
        print("========== include korquad ==========")
        korquad_dataset = load_dataset("squad_kor_v1")

        total = []
        for idx, example in enumerate(tqdm(datasets['train'], desc="Concatenate klue: ")):
            tmp = {
                "question": example["question"],
                "id": example['id'],
                "context": example['context'],
                "answers": example['answers'],  # original answer
                "title": example['title'],  # original answer
            }
            total.append(tmp)

        for idx, example in enumerate(tqdm(korquad_dataset['train'], desc="Concatenate korquad: ")):
            if len(example['context']) >= 500:
                tmp = {
                    "question": example["question"],
                    "id": example['id'],
                    "context": example['context'],
                    "answers": example['answers'],  # original answer
                    "title": example['title'],  # original answer
                }
                total.append(tmp)

        datasets['train'] = Dataset.from_pandas(pd.DataFrame(total))

        total = []
        for idx, example in enumerate(tqdm(datasets['validation'], desc="Concatenate klue validation: ")):
            tmp = {
                "question": example["question"],
                "id": example['id'],
                "context": example['context'],
                "answers": example['answers'],  # original answer
                "title": example['title'],  # original answer
            }
            total.append(tmp)

        datasets['validation'] = Dataset.from_pandas(pd.DataFrame(total))
        print(datasets)
        # exit()

    # train & save sparse embedding retriever if true
    if data_args.train_retrieval:
        datasets['train'] = run_sparse_embedding_train(datasets['train'], training_args, data_args)
        print("============check==============")
        # exit()

    if data_args.eval_retrieval:
        datasets['validation'] = run_sparse_embedding_val(datasets['validation'], training_args, data_args)
        print("============check==============")
        # exit()


def run_sparse_embedding_train(datasets, training_args, data_args):
    retriever = SparseRetrieval(tokenize_fn=tokenize,
                                data_path="/opt/ml/input/data",
                                context_path="wikipedia_documents.json",
                                args=data_args)

    if data_args.embedding_mode != 'bm25_new':
        retriever.get_sparse_embedding()

    # if training_args.do_train: # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    df = retriever.retrieve(datasets, topk=data_args.topk, what="train_train")
    f = Features({'answers': Sequence(feature={'text': Value(dtype='string', id=None),
                                               'answer_start': Value(dtype='int32', id=None)},
                                      length=-1, id=None),
                  'context': Value(dtype='string', id=None),
                  'id': Value(dtype='string', id=None),
                  'question': Value(dtype='string', id=None)})

    # datasets = DatasetDict({'train': Dataset.from_pandas(df, features=f)})
    datasets = Dataset.from_pandas(df, features=f)

    print(datasets)
    return datasets


def run_sparse_embedding_val(datasets, training_args, data_args):
    retriever = SparseRetrieval(tokenize_fn=tokenize,
                                data_path="/opt/ml/input/data",
                                context_path="wikipedia_documents.json",
                                args=data_args)

    if data_args.embedding_mode != 'bm25_new':
        retriever.get_sparse_embedding()

    df = retriever.retrieve(datasets, topk=data_args.topk, what="train_val")
    f = Features({'answers': Sequence(feature={'text': Value(dtype='string', id=None),
                                               'answer_start': Value(dtype='int32', id=None)},
                                      length=-1, id=None),
                  'context': Value(dtype='string', id=None),
                  'id': Value(dtype='string', id=None),
                  'question': Value(dtype='string', id=None)})

    # datasets = DatasetDict({'validation': Dataset.from_pandas(df, features=f)})
    datasets = Dataset.from_pandas(df, features=f)

    print(datasets)
    return datasets


if __name__ == "__main__":
    main()
