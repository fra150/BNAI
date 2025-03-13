# Nelle importazioni, assicurati di usare percorsi relativi al progetto
import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from src.utils.config import load_config  # Usa il percorso relativo corretto

def load_sentiment_labels(file_path):
    """Carica le etichette di sentiment dal file"""
    sentiment_labels = {}
    with open(file_path, 'r') as f:
        # Salta l'intestazione
        next(f)
        for line in f:
            phrase_id, score = line.strip().split('|')
            sentiment_labels[phrase_id] = float(score)
    return sentiment_labels

def load_dictionary(file_path):
    """Carica il dizionario delle frasi"""
    dictionary = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 2:
                phrase, phrase_id = parts
                dictionary[phrase_id] = phrase
    return dictionary

def generate_bnai_profile(sentiment_score, length, complexity):
    """
    Genera un profilo BNAI sintetico basato sul sentiment
    
    Args:
        sentiment_score: Punteggio di sentiment (0-1)
        length: Lunghezza della frase
        complexity: Complessità sintattica stimata
    """
    # Normalizza i valori
    sentiment_norm = sentiment_score * 2 - 1  # Da 0-1 a -1-1
    length_norm = min(length / 100, 1.0)      # Normalizza lunghezza
    
    # Genera parametri BNAI basati sul sentiment e altre caratteristiche
    profile = {
        'A': 0.8 + 0.4 * sentiment_norm,                # Adaptability
        'E_g': 0.9 + 0.2 * random.random(),             # Evolution
        'G': 1.5 + length_norm,                         # Growth
        'H': 1.0 + 0.3 * complexity,                    # Learning Entropy
        'I': 0.6 + 0.2 * sentiment_norm,                # Interconnection
        'L': 1.2 + 0.3 * complexity,                    # Learning Level
        'P': 0.8 + 0.2 * length_norm,                   # Processing Power
        'Q': 2.0 + sentiment_norm,                      # Quality
        'R': 0.7 + 0.25 * abs(sentiment_norm),          # Robustness
        'U': 0.9 + 0.2 * random.random(),               # Autonomy
        'V': 0.7 + 0.2 * (1 - length_norm),             # Speed
        'O': 0.8 + 0.2 * random.random(),               # Optimization
        'T': 1.0 + 0.2 * random.random(),               # Time Factor
        'S': 0.8 + 0.2 * (1 - abs(sentiment_norm)),     # Stability
        'F': 0.7 + 0.15 * random.random(),              # Flexibility
        'B': 0.3 + 0.2 * (1 - sentiment_norm),          # Bias
        'C': 0.2 + 0.1 * complexity,                    # Complexity
        'E': 0.1 + 0.1 * (1 - sentiment_norm),          # Error Rate
        'M': 3.0 + 2.0 * length_norm,                   # Memory
        't': 1.0 + random.random(),                     # Time
        'D': 0.6 + 0.15 * random.random(),              # Durability
        'K': 0.7 + 0.15 * sentiment_norm,               # Knowledge Retention
        'Z': 0.6 + 0.1 * random.random(),               # Self-optimization
    }
    
    # Assicurati che i valori siano nei range corretti
    for key, value in profile.items():
        if key == 'G':
            profile[key] = min(max(value, 0.0), 3.0)
        elif key == 'Q':
            profile[key] = min(max(value, 0.0), 3.0)
        elif key == 'M':
            profile[key] = min(max(value, 0.0), 10.0)
        elif key == 't':
            profile[key] = min(max(value, 0.0), 5.0)
        else:
            profile[key] = min(max(value, 0.0), 2.0)
    
    return profile

def convert_stanford_to_bnai(stanford_dir, output_file):
    """
    Converte il dataset Stanford Sentiment Treebank in un dataset BNAI
    """
    try:
        # Carica le etichette di sentiment
        sentiment_labels_path = os.path.join(stanford_dir, 'sentiment_labels.txt')
        sentiment_labels = load_sentiment_labels(sentiment_labels_path)
        
        # Carica il dizionario
        dictionary_path = os.path.join(stanford_dir, 'dictionary.txt')
        dictionary = load_dictionary(dictionary_path)
        
        # Carica le frasi
        sentences_path = os.path.join(stanford_dir, 'datasetSentences.txt')
        sentences = []
        with open(sentences_path, 'r', encoding='utf-8') as f:
            next(f)  # Salta l'intestazione
            for line in f:
                if '\t' in line:
                    sentence_id, sentence = line.strip().split('\t')
                    sentences.append((sentence_id, sentence.strip()))
        
        # Genera profili BNAI
        bnai_data = []
        for sentence_id, sentence in tqdm(sentences, desc="Generando profili BNAI"):
            # Cerca la frase nel dizionario
            phrase_id = None
            for pid, phrase in dictionary.items():
                if phrase.strip() == sentence.strip():
                    phrase_id = pid
                    break
            
            if phrase_id and phrase_id in sentiment_labels:
                sentiment_score = sentiment_labels[phrase_id]
                length = len(sentence.split())
                complexity = min(1.0, length / 30)
                
                bnai_profile = generate_bnai_profile(
                    sentiment_score=sentiment_score,
                    length=length,
                    complexity=complexity
                )
                
                bnai_data.append({
                    'sentence': sentence,
                    'sentiment': sentiment_score,
                    'bnai_profile': bnai_profile
                })
        
        # Salva il dataset BNAI
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(bnai_data, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset BNAI creato con {len(bnai_data)} esempi")
        return bnai_data
        
    except Exception as e:
        print(f"Errore durante la conversione: {str(e)}")
        return []

if __name__ == "__main__":
    stanford_dir = "c:\\Users\\acese\\Desktop\\BNAI\\src\\stanfordSentimentTreebank"
    output_file = "c:\\Users\\acese\\Desktop\\BNAI\\data\\synthetic_bnai_data.json"
    
    # Crea la directory di output se non esiste
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Converti il dataset
    convert_stanford_to_bnai(stanford_dir, output_file)