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
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    return df

if __name__ == '__main__':
    input_file = '/Users/tukyy/Downloads/Git/working_home/toi_uu_hoa/optimzer_2025/data/IMDB Dataset.csv'
    output_file = '/Users/tukyy/Downloads/Git/working_home/toi_uu_hoa/optimzer_2025/data/processed_imdb.csv'
    
    df = load_data(input_file)
    df.to_csv(output_file, index=False)
    
    print(f"Processed data saved to {output_file}")
    print(df.head())