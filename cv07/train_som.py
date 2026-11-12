# train_som.py
import torch
import numpy as np

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Používám zařízení: {device}")

    print("Načítání vektorů Wikipedie...")
    X_np = np.load("wiki_embeddings.npy").astype(np.float32)  # (241787, 384)
    X = torch.from_numpy(X_np).to(device)
    
    N, D = X.shape
    G = 40
    epochs = 3
    T = N * epochs

    radek = torch.arange(G * G, device=device) // G
    sloupec = torch.arange(G * G, device=device) % G

    perm_init = torch.randperm(N, device=device)[:G * G]
    W = X[perm_init].clone()

    print(f"Start trénování SOM ({G}x{G}, {epochs} epoch, T={T} kroků)...")

    t = 0
    for epoch in range(epochs):
        perm_epoch = torch.randperm(N, device=device)
        for idx in perm_epoch:
            x_j = X[idx]

            dists_sq = torch.sum((W - x_j) ** 2, dim=1)
            best = torch.argmin(dists_sq)

            t_ratio = t / T
            alpha = 0.5 + (0.02 - 0.5) * t_ratio
            beta = alpha / 2.0
            r = (G / 2.0) * (1.0 - t_ratio)

            region = ((radek - radek[best]).abs() <= r) & ((sloupec - sloupec[best]).abs() <= r)
            region[best] = False

            W[best] += alpha * (x_j - W[best])
            if torch.any(region):
                W[region] += beta * (x_j - W[region])

            t += 1

        print(f"Epocha {epoch + 1}/{epochs} dokončena.")

    W_cpu = W.cpu().numpy()
    np.save("som_W.npy", W_cpu)
    print("Váhy úspěšně uloženy do som_W.npy")

if __name__ == "__main__":
    train()