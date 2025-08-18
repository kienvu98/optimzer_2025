from bs4 import BeautifulSoup
import pandas as pd
import re

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
    

        