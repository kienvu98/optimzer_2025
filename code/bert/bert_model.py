from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelBertEncoder():
    
    def __init__(self, model_name = "bert-base-uncased", device = DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval() 
        print(f"BERT run with: {self.device}")
        
    def encode(self, text_list):
        inputs = self.tokenizer(text_list, return_tensors="pt", truncation=True, padding=True, max_length=256).to(self.device)
        
        # lấy output
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        cls_vector = outputs.last_hidden_state[:, 0, :]
        return cls_vector.cpu().numpy()
        
    