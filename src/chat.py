import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import json
import torch
import numpy as np

class BNAIChat:
    def __init__(self, data_path="data/bnai_data.json"):
        self.data = self.load_data(data_path)
        self.conversation_history = []
        
    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_bnai_response(self, user_input):
        best_match = None
        best_score = -1
        
        for item in self.data:
            score = self.simple_similarity(user_input.lower(), item['sentence'].lower())
            if score > best_score:
                best_score = score
                best_match = item
        
        if best_match:
            profile = best_match['bnai_profile']
            sentiment = best_match['sentiment']
            original_sentence = best_match['sentence']
            
            # Prima genera una risposta contestuale al contenuto
            content_response = self.generate_content_response(user_input, original_sentence, profile)
            
            # Poi aggiunge l'analisi BNAI
            response = f"{content_response}\n\nAnalisi BNAI dei parametri utilizzati:\n"
            response += f"- Comprensione: {profile['A']:.2f}/2.0\n"
            response += f"- Elaborazione: {profile['G']:.2f}/3.0\n"
            response += f"- Conoscenza: {profile['K']:.2f}/1.0\n"
            response += f"- Sentiment: {sentiment:.2f}/1.0\n"
            
            if profile['G'] > 1.5:
                response += "\nPosso esplorare ulteriori prospettive se lo desideri."
            
            return response
        
        return "Mi dispiace, non ho sufficienti informazioni per rispondere adeguatamente. Puoi riformulare la domanda?"

    def generate_content_response(self, user_input, reference_text, profile):
        # Genera una risposta basata sul contenuto e modulata dai parametri BNAI
        creativity = profile['G']
        knowledge = profile['K']
        
        # Analizza il tipo di input
        if "?" in user_input:
            if creativity > 1.8:
                return f"La tua domanda è interessante e mi fa pensare a diverse possibilità. Considerando la mia base di conoscenza, suggerirei che {reference_text}"
            else:
                return f"Basandomi sulla mia analisi, {reference_text}"
        
        elif any(word in user_input.lower() for word in ["cosa pensi", "opinione", "parere"]):
            if knowledge > 0.8:
                return f"Ho una prospettiva ben definita su questo. {reference_text}"
            else:
                return f"Dal mio punto di vista, anche se con alcune incertezze, {reference_text}"
        
        else:
            if creativity > 1.5:
                return f"Questo mi fa pensare a un'interessante connessione: {reference_text}"
            else:
                return f"La mia analisi suggerisce che {reference_text}"

    def simple_similarity(self, text1, text2):
        # Simple word overlap similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0

def main():
    chat = BNAIChat()
    print("BNAI: Ciao! Sono pronto per chattare. Scrivi 'exit' per terminare.")
    
    while True:
        user_input = input("Tu: ").strip()
        if user_input.lower() == 'exit':
            print("BNAI: Arrivederci! È stato un piacere conversare con te.")
            break
            
        response = chat.get_bnai_response(user_input)
        print(f"BNAI: {response}")

if __name__ == "__main__":
    main()