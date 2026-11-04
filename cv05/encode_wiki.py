import os
import time
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def main():
    print("Načítám Simple Wikipedii (20231101.simple)...")

    start_time = time.time()
    dataset = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    
    titles = dataset["title"]
    texts = dataset["text"]
    
    full_texts = [f"{t}. {txt[:2000]}" for t, txt in zip(titles, texts)]
    print(f"Načteno {len(full_texts)} článků.")

    print("Načítám model multi-qa-MiniLM-L6-cos-v1")
    model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", device="cuda")

    print("Počítám vektory")
    embeddings = model.encode(full_texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)

    np.save("wiki_embeddings.npy", embeddings.astype(np.float16))
    end_time = time.time()
    print(f"Soubor wiki_embeddings.npy byl úspěšně vytvořen. Čas trvání: {end_time - start_time:.2f} s.")

if __name__ == "__main__":
    main()