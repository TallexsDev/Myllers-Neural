import os
import numpy as np

from core.tokenizer import Tokenizer
from core.model import MyllersCore

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "train.txt")
MODELS = os.path.join(BASE, "models")

os.makedirs(MODELS, exist_ok=True)

with open(DATA, encoding="utf-8") as f:
    text = f.read()

if len(text) < 100:
    raise ValueError("Adicione mais textos ao arquivo data/train.txt.")

tokenizer = Tokenizer()
tokenizer.train(text)

ids = np.array(tokenizer.encode(text), dtype=np.int64)

model = MyllersCore(
    vocab_size=tokenizer.vocab_size,
    context=16,
    embedding=24,
    hidden=64
)

tokenizer.save(os.path.join(MODELS, "tokenizer.json"))

print("================================")
print("       MYLLERS NEURAL CORE")
print("================================")
print("Vocabulário:", tokenizer.vocab_size)
print("Caracteres:", len(ids))
print("Parâmetros:", sum(p.size for p in model.params.values()))
print("Iniciando treinamento...")

rng = np.random.default_rng(42)
epochs = 1500
batch_size = 32

for step in range(1, epochs + 1):
    positions = rng.integers(16, len(ids), size=batch_size)

    x = np.stack([ids[p-16:p] for p in positions])
    y = ids[positions]

    loss = model.train_step(x, y)

    if step % 100 == 0:
        print(f"Passo {step}/{epochs} | Loss: {loss:.4f}")

model.save(os.path.join(MODELS, "myllers_core.npz"))

print("\nTreinamento concluído!")
print("Pesos salvos em models/myllers_core.npz")
