import numpy as np

import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# 질문 입력
query = str(input())

#모델 로드
MODEL_NAME = "bert-base-multilingual-cased"
tokenizer = torch.load("./tokenizer")
model = torch.load("./chat_bot_bert_model")
dataset_cls_hidden = np.load('./ChatBot_numpy_model.npy')
chatbot_Answer = np.load('./Answer_data.npy')

def get_cls_token(sent_A):
    model.eval()
    tokenized_sent = tokenizer(
            sent_A,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=128
    )
    with torch.no_grad():# 그라디엔트 계산 비활성화
        outputs = model(    # **tokenized_sent
            input_ids=tokenized_sent['input_ids'],
            attention_mask=tokenized_sent['attention_mask'],
            token_type_ids=tokenized_sent['token_type_ids']
            )
    logits = outputs.last_hidden_state[:,0,:].detach().cpu().numpy()
    return logits

query_cls_hidden = get_cls_token(query)

cos_sim = cosine_similarity(query_cls_hidden, dataset_cls_hidden)

top_question = np.argmax(cos_sim)

print('나의 질문: ', query)
print('저장된 답변: ', chatbot_Answer[top_question])



