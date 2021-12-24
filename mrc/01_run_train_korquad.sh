python /home/koo/ref_source/transformers-v4.2.2/examples/question-answering/run_qa.py \
--model_name_or_path bert-base-multilingual-cased  \
--output_dir /home/koo/source/temp2/models/MBERT-korquad  \
--dataset_name squad_kor_v2  \
--do_train  \
--do_eval \
--num_train_epochs 3  \
--overwrite_output_dir


