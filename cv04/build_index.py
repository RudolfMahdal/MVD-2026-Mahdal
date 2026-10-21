import argparse
import time
import re
import pickle
import spacy
from datasets import load_dataset
from collections import defaultdict, Counter

def preprocess_batch(texts, nlp, n_process):
    """Předzpracuje dávku textů paralelně přes spaCy."""
    # 1. a 2. krok: Malá písmena a odstranění speciálních znaků (rychlejší v čistém Pythonu)
    cleaned_texts = [re.sub(r"[^\w\s]", " ", text.lower()) if text else "" for text in texts]
    
    processed = []
    # 3. krok: Lemmatizace přes nlp.pipe
    for doc in nlp.pipe(cleaned_texts, batch_size=200, n_process=n_process):
        processed.append([token.lemma_ for token in doc if not token.is_space])
    return processed

def build_index_with_counts(processed_docs):
    """Vytvoří invertovaný index s počty výskytů c(w,d)."""
    index = defaultdict(dict)
    for doc_id, tokens in enumerate(processed_docs):
        counts = Counter(tokens)
        for word, count in counts.items():
            index[word][doc_id] = count
    return dict(index)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print("Načítám dataset...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    
    if args.limit:
        dataset = dataset.select(range(args.limit))
    
    titles = dataset["title"]
    texts = dataset["text"]
    
    print(f"Spouštím spaCy na {args.cores} jádrech pro {len(dataset)} dokumentů...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
    start_time = time.time()
    
    prep_titles = preprocess_batch(titles, nlp, args.cores)
    prep_texts = preprocess_batch(texts, nlp, args.cores)
    
    elapsed = time.time() - start_time
    print(f"Předzpracování dokončeno za {elapsed:.2f} sekund.")

    print("Vytvářím indexy...")
    title_index = build_index_with_counts(prep_titles)
    text_index = build_index_with_counts(prep_texts)

    print("Ukládám data na disk...")
    with open("wiki_indices.pkl", "wb") as f:
        pickle.dump({
            "title_index": title_index,
            "text_index": text_index,
            "doc_count": len(dataset),
            "titles_raw": titles 
        }, f)
    print("Hotovo.")

if __name__ == "__main__":
    main()
