import numpy as np


class MyllersCore:
    def __init__(self, vocab_size, context=16, embedding=24, hidden=64):
        self.vocab_size = vocab_size
        self.context = context
        self.embedding = embedding
        self.hidden = hidden

        rng = np.random.default_rng()

        self.params = {
            "E": rng.normal(0, 0.08, (vocab_size, embedding)).astype(np.float32),
            "W1": (rng.normal(0, 1, (context * embedding, hidden))
                   * np.sqrt(2 / (context * embedding))).astype(np.float32),
            "b1": np.zeros(hidden, dtype=np.float32),
            "W2": (rng.normal(0, 0.02, (hidden, vocab_size))
                   ).astype(np.float32),
            "b2": np.zeros(vocab_size, dtype=np.float32),
        }

        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.step = 0

    def forward(self, x):
        emb = self.params["E"][x]
        flat = emb.reshape(len(x), -1)

        z = flat @ self.params["W1"] + self.params["b1"]
        h = np.maximum(z, 0)

        logits = h @ self.params["W2"] + self.params["b2"]
        logits -= logits.max(axis=1, keepdims=True)

        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)

        return probs, (x, flat, z, h)

    def train_step(self, x, y, lr=0.001):
        probs, cache = self.forward(x)
        x, flat, z, h = cache

        batch = len(x)
        loss = -np.log(probs[np.arange(batch), y] + 1e-9).mean()

        dlogits = probs.copy()
        dlogits[np.arange(batch), y] -= 1
        dlogits /= batch

        grads = {}

        grads["W2"] = h.T @ dlogits
        grads["b2"] = dlogits.sum(axis=0)

        dh = dlogits @ self.params["W2"].T
        dz = dh * (z > 0)

        grads["W1"] = flat.T @ dz
        grads["b1"] = dz.sum(axis=0)

        demb = dz @ self.params["W1"].T
        demb = demb.reshape(batch, self.context, self.embedding)

        grads["E"] = np.zeros_like(self.params["E"])
        np.add.at(grads["E"], x.reshape(-1), demb.reshape(-1, self.embedding))

        self.step += 1
        beta1, beta2 = 0.9, 0.999

        for key in self.params:
            g = np.clip(grads[key], -1.0, 1.0)

            self.m[key] = beta1 * self.m[key] + (1 - beta1) * g
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (g * g)

            m_hat = self.m[key] / (1 - beta1 ** self.step)
            v_hat = self.v[key] / (1 - beta2 ** self.step)

            self.params[key] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        return float(loss)

    def predict(self, context_ids, temperature=0.8):
        x = np.array([context_ids[-self.context:]], dtype=np.int64)
        probs, _ = self.forward(x)

        logits = np.log(probs[0] + 1e-9) / max(temperature, 0.05)
        logits -= logits.max()

        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()

        return int(np.random.choice(self.vocab_size, p=probabilities))

    def save(self, path):
        np.savez(path, **self.params)

    def load(self, path):
        data = np.load(path)
        for key in self.params:
            self.params[key] = data[key].astype(np.float32)
