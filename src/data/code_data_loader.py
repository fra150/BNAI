import json
import csv
import gzip
import torch
from torch.utils.data import Dataset, DataLoader
import os

class BNAICodeDataset(Dataset):
    """
    Dataset class for BNAI code data, supporting both CSV and JSONL.GZ formats
    """
    def __init__(self, data, tokenizer=None, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract problem and solution code
        problem = item.get('problem', '')
        
        # Handle case where solution_code is an empty list or not present
        solution_code_raw = item.get('solution_code', '')
        # If solution_code is a list, join it or use empty string if empty
        if isinstance(solution_code_raw, list):
            solution_code = '\n'.join(solution_code_raw) if solution_code_raw else ''
        else:
            solution_code = str(solution_code_raw)
        
        # If tokenizer is provided, tokenize the text
        if self.tokenizer:
            problem_encoding = self.tokenizer(problem, 
                                           max_length=self.max_length,
                                           padding='max_length',
                                           truncation=True,
                                           return_tensors='pt')
            
            solution_encoding = self.tokenizer(solution_code,
                                            max_length=self.max_length,
                                            padding='max_length',
                                            truncation=True,
                                            return_tensors='pt')
            
            return {
                'problem_ids': problem_encoding['input_ids'].squeeze(),
                'problem_mask': problem_encoding['attention_mask'].squeeze(),
                'solution_ids': solution_encoding['input_ids'].squeeze(),
                'solution_mask': solution_encoding['attention_mask'].squeeze()
            }
        else:
            # Return raw text if no tokenizer
            return {
                'problem': problem,
                'solution_code': solution_code
            }

def load_csv_data(file_path):
    """
    Load data from CSV file
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def load_jsonl_gz_data(file_path):
    """
    Load data from JSONL.GZ file
    """
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    continue
    return data

def load_bnai_code_data(data_path, batch_size=32, tokenizer=None, max_length=512):
    """
    Load BNAI code data from either CSV or JSONL.GZ file
    """
    try:
        # Determine file type and load accordingly
        if data_path.endswith('.csv'):
            data = load_csv_data(data_path)
        elif data_path.endswith('.jsonl.gz'):
            data = load_jsonl_gz_data(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        if not data:
            raise ValueError(f"No data found in {data_path}")
            
        # Create dataset
        dataset = BNAICodeDataset(data, tokenizer, max_length)
        
        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Loaded {len(data)} examples from {data_path}")
        
        return dataloader
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {data_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def load_train_val_test_data(train_path, val_path=None, test_path=None, batch_size=32, tokenizer=None, max_length=512):
    """
    Load train, validation, and test data
    """
    train_loader = load_bnai_code_data(train_path, batch_size, tokenizer, max_length)
    
    val_loader = None
    if val_path and os.path.exists(val_path):
        val_loader = load_bnai_code_data(val_path, batch_size, tokenizer, max_length)
    
    test_loader = None
    if test_path and os.path.exists(test_path):
        test_loader = load_bnai_code_data(test_path, batch_size, tokenizer, max_length)
    
    return train_loader, val_loader, test_loader