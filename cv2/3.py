import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE

X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True, parser="liac-arff")
X, y = X[:30000], y[:30000]
start = time.time()

# tsne vypocet
tsne = TSNE(n_components=2, method="exact", random_state=42)
X_2d = tsne.fit_transform(X)

print(f"Elapsed time: {time.time() - start:.2f} s")

plt.figure(figsize=(10, 8))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y.astype(int), cmap='tab10', s=2)
plt.colorbar()
plt.savefig("mnist_tsne.png")
