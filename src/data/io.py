import json
import torch
from torch.utils.data import Dataset, DataLoader
from .preprocess import preprocess_bnai_data

class BNAIDataset(Dataset):
    def __init__(self, data):
        # Separa features e target
        self.features = torch.tensor([list(item['bnai_profile'].values()) for item in data], dtype=torch.float32)
        self.targets = self.features.clone()  # Per l'autoencoder, target = input
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def load_bnai_data(file_path, train_split=0.8, batch_size=32):
    """
    Carica i dati BNAI dal file JSON e li divide in train e validation
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not data:
            raise ValueError(f"Il file {file_path} non contiene dati validi")
            
        # Calcola la divisione train/validation
        split_idx = int(len(data) * train_split)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # Crea i dataset
        train_dataset = BNAIDataset(train_data)
        val_dataset = BNAIDataset(val_data)
        
        # Crea i DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        print(f"Dati caricati: {len(train_data)} esempi di training, {len(val_data)} esempi di validazione")
        
        return train_loader, val_loader
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File non trovato: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Errore nel parsing del file JSON: {file_path}")
    except Exception as e:
        raise Exception(f"Errore nel caricamento dei dati: {str(e)}")