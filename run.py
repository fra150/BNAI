import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data.convert_stanford_to_bnai import convert_stanford_to_bnai
from src.data.synthetic_data import generate_synthetic_bnai_data
from src.experiments.run_experiment import run_experiment

if __name__ == "__main__":
    # Percorsi
    stanford_dir = "c:\\Users\\acese\\Desktop\\BNAI\\src\\stanfordSentimentTreebank"
    bnai_data_path = "c:\\Users\\acese\\Desktop\\BNAI\\data\\bnai_data.json"
    
    # Crea directory se non esistono
    os.makedirs("c:\\Users\\acese\\Desktop\\BNAI\\data", exist_ok=True)
    os.makedirs("c:\\Users\\acese\\Desktop\\BNAI\\models", exist_ok=True)
    os.makedirs("c:\\Users\\acese\\Desktop\\BNAI\\checkpoints", exist_ok=True)
    
    # Converti il dataset Stanford in dati BNAI
    print("Conversione del dataset Stanford in dati BNAI...")
    convert_stanford_to_bnai(stanford_dir, bnai_data_path)
    
    # Se non ci sono abbastanza dati, genera dati sintetici aggiuntivi
    if not os.path.exists(bnai_data_path) or os.path.getsize(bnai_data_path) < 1000:
        print("Generazione di dati sintetici aggiuntivi...")
        generate_synthetic_bnai_data(num_samples=2000, output_file=bnai_data_path)
    
    # Esegui l'esperimento
    print("Avvio dell'esperimento BNAI...")
    run_experiment(config_path="c:\\Users\\acese\\Desktop\\BNAI\\src\\utils\\config.yaml")