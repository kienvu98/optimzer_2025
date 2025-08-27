from bs4 import BeautifulSoup
import pandas as pd
from optimzer_project.code.bert.bert_model import ModelBertEncoder
import re
from tqdm import tqdm
import numpy as np
import os

def clean_data(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)  # bỏ ký tự điều khiển
    text = re.sub(r'\s+', ' ', text)  # chuẩn hóa khoảng trắng
    
    text = re.sub(r"[^a-zA-Z0-9\s\.,!?'\"]", " ", text)
    text = text.lower().strip()
    
    # Lowercase nếu dùng bert-base-uncased
    text = text.lower().strip()
    
    return text


def load_data(file_name):
    df = pd.read_csv(file_name)
    df['review'] = df['review'].astype(str).apply(clean_data)
    df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})
    texts = df['review'].tolist()
    labels = df['sentiment'].values.reshape(-1,1)
    return texts, labels

# encoder bằng BERT
def encode_bert(texts, batch_size=128):
    model_bert = ModelBertEncoder()
    all_vector = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding BERT"):
        batch_texts = texts[i:i+batch_size]
       # cls_vector = model_bert.encode(batch_texts)
        all_vector.append(model_bert.encode(batch_texts))
    
    return np.vstack(all_vector)
        
    
# Pipeline xử lý và lưu
def process_and_save(file_name, output_dir, output_prefix="imdb_encoded", batch_size=128):
    texts, labels = load_data(file_name)
    X = encode_bert(texts, batch_size=batch_size)
    y = labels.astype(np.int32)
    path_X = os.path.join(output_dir, f"{output_prefix}_X.npy")
    path_y = os.path.join(output_dir, f"{output_prefix}_y.npy")
    np.save(path_X, X)
    np.save(path_y, y)
    print(f"✅ Đã lưu: {output_prefix}_X.npy và {output_prefix}_y.npy")
    
def main():
    file_name = '/workspace/optimzer_project/data/IMDB Dataset.csv'
    output_dir = 'data/data_optimzer_project_train'
    process_and_save(file_name=file_name, output_dir=output_dir)


if __name__ == "__main__":
    main()