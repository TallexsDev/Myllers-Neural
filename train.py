import os
import numpy as np

from core.tokenizer import Tokenizer
from core.model import MyllersCore

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "train.txt")
MODELS = os.path.join(BASE, "models")

os.makedirs(MODELS, exist_ok=True)

with open(DATA, encoding="utf-8") as file:
    text = file.read()

tokenizer = Tokenizer()
tokenizer.train(text)

ids = np.array(tokenizer.encode(text), dtype=np.int64)

context = 32
embedding = 24
hidden = 48

split = int(len(ids) * 0.90)

train_ids = ids[:split]
val_ids = ids[split:]

if len(val_ids) <= context or len(train_ids) <= context:
    raise ValueError("Dataset insuficiente.")

rng = np.random.default_rng(42)

model = MyllersCore(
    vocab_size=tokenizer.vocab_size,
    context=context,
    embedding=embedding,
    hidden=hidden
)

tokenizer.save(
    os.path.join(MODELS, "tokenizer.json")
)

def get_batch(source, batch_size):
    positions = rng.integers(
        0,
        len(source) - context,
        size=batch_size
    )

    x = np.stack([
        source[p:p + context]
        for p in positions
    ])

    y = np.stack([
        source[p + 1:p + context + 1]
        for p in positions
    ])

    return x, y


def evaluate(source):
    x, y = get_batch(source, 4)
    probs, _ = model.forward(x)

    selected = probs[
        np.arange(len(y))[:, None],
        np.arange(context)[None, :],
        y
    ]

    return float(-np.log(selected + 1e-9).mean())


print("=" * 48)
print("          MYLLERS NEURAL")
print("          CORE VERSION 0.3")
print("=" * 48)

print("Criador: Tallexs")
print("Arquitetura: Transformer causal")
print("Cabeças: 1")
print("Contexto:", context)
print("Embedding:", embedding)
print("Camada oculta:", hidden)
print("Vocabulário:", tokenizer.vocab_size)
print("Caracteres:", len(ids))
print("Parâmetros:", sum(p.size for p in model.params.values()))

epochs = 2000
batch_size = 4

print("\nIniciando treinamento do zero...\n")

for step in range(1, epochs + 1):
    x, y = get_batch(train_ids, batch_size)

    loss = model.train_step(
        x,
        y,
        lr=0.0005
    )

    if step % 100 == 0:
        val_loss = evaluate(val_ids)

        print(
            f"Passo {step}/{epochs} | "
            f"Treino: {loss:.4f} | "
            f"Validação: {val_loss:.4f}"
        )

    if step % 500 == 0:
        model.save(
            os.path.join(MODELS, "myllers_checkpoint.npz")
        )

model.save(
    os.path.join(MODELS, "myllers_neural_v03.npz")
)

print("\n================================")
print("TREINAMENTO FINALIZADO")
print("Criador: Tallexs")
print("Modelo: myllers_neural_v03.npz")
print("================================")
