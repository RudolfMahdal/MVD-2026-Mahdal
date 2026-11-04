import gzip
import json
import time
import numpy as np
import torch
from bs4 import BeautifulSoup
from transformers import AutoModel, AutoProcessor


def clean_html(html_str):
    soup = BeautifulSoup(html_str, "lxml")
    for tag in soup(["nav", "header", "footer", "aside", "form", "script", "style"]):
        tag.decompose()
    main_tag = soup.find(["main", "article"])
    res = str(main_tag) if main_tag else (str(soup.body) if soup.body else str(soup))
    return res if res.strip() else "<html><body></body></html>"


def extract_and_save(pages_data, prefix, processor, model, batch_size=16):
    start_time = time.time()

    html_texty = [p["html"] for p in pages_data]
    y = np.array([p["category"] for p in pages_data])
    splits = np.array([p["split"] for p in pages_data])

    print(f"\n--- Processing: {prefix} ---", flush=True)
    print("Tokenizing HTML texts...", flush=True)
    ids = processor(html_texty, truncation=True, max_length=4096)["input_ids"]

    over_limit = sum(1 for item in ids if len(item) >= 4096)
    under_limit = len(pages_data) - over_limit
    print(f"Pages fitting into 4096 tokens: {under_limit} / {len(pages_data)} ({under_limit / len(pages_data) * 100:.2f}%)", flush=True)
    print(f"Pages truncated at 4096 tokens: {over_limit} / {len(pages_data)} ({over_limit / len(pages_data) * 100:.2f}%)", flush=True)

    # Seřazení podle počtu TOKENŮ (minimalizuje padding v dávce) + uložení indexů pro vrácení do původního pořadí
    sorted_indices = np.argsort([len(seq) for seq in ids])
    sorted_ids = [ids[i] for i in sorted_indices]

    vektory_sorted = []
    use_bf16 = True

    print("Extracting vectors using HTML-LM on GPU...", flush=True)
    with torch.no_grad():
        for i in range(0, len(sorted_ids), batch_size):
            batch_ids = sorted_ids[i : i + batch_size]
            x = processor.tokenizer.pad({"input_ids": batch_ids}, return_tensors="pt").to("cuda")

            if use_bf16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).last_hidden_state
                vektory = out[:, 0].float()

                # Pojistka: pokud Longformer v bfloat16 vyrobí NaN kvůli paddingu, přepneme na čistý float32
                if torch.isnan(vektory).any():
                    print(f"WARNING: NaN detected in bfloat16 at batch {i}! Switching to float32 for safety...", flush=True)
                    use_bf16 = False
                    out = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).last_hidden_state
                    vektory = out[:, 0].float()
            else:
                out = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).last_hidden_state
                vektory = out[:, 0].float()

            vektory_sorted.append(vektory.cpu().numpy())

    X_sorted = np.concatenate(vektory_sorted, axis=0)

    # Vrácení vektorů do původního pořadí stránek
    X = np.empty_like(X_sorted)
    X[sorted_indices] = X_sorted

    nan_count = np.isnan(X).any(axis=1).sum()
    print(f"Final NaN vectors check: {nan_count} / {len(X)}", flush=True)

    np.savez_compressed(f"{prefix}.npz", X=X, y=y, splits=splits)

    elapsed_time = time.time() - start_time
    print(f"Vectors successfully saved to {prefix}.npz with shape {X.shape}", flush=True)
    print(f"Extraction time for {prefix}: {elapsed_time:.2f} s ({elapsed_time / 60:.2f} min)", flush=True)

    return elapsed_time


def main():
    total_start_time = time.time()

    print("Loading dataset from curlie_cs.jsonl.gz...", flush=True)
    raw_pages = [json.loads(line) for line in gzip.open("curlie_cs.jsonl.gz", "rt", encoding="utf-8")]
    print(f"Loaded {len(raw_pages)} pages.", flush=True)

    print("Loading Seznam/html-lm model and processor to CUDA...", flush=True)
    processor = AutoProcessor.from_pretrained("Seznam/html-lm", trust_remote_code=True)
    model = AutoModel.from_pretrained("Seznam/html-lm").to("cuda").eval()

    # 1. Raw HTML
    pages_raw = [dict(p) for p in raw_pages]
    time_raw = extract_and_save(pages_raw, "vectors_raw", processor, model)

    # 2. Cleaned HTML
    print("\nCleaning HTML tags (nav, header, footer, aside, form, etc.)...", flush=True)
    clean_prep_start = time.time()
    pages_clean = []
    for p in raw_pages:
        p_copy = dict(p)
        p_copy["html"] = clean_html(p_copy["html"])
        pages_clean.append(p_copy)
    clean_prep_time = time.time() - clean_prep_start
    print(f"HTML cleaning preprocessing time: {clean_prep_time:.2f} s", flush=True)

    time_clean_gpu = extract_and_save(pages_clean, "vectors_clean", processor, model)

    total_elapsed = time.time() - total_start_time
    print(f"\nTotal script execution time: {total_elapsed:.2f} s ({total_elapsed / 60:.2f} min)", flush=True)


if __name__ == "__main__":
    main()