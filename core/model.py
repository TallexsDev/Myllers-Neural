import numpy as np


class MyllersCore:
    def __init__(
        self,
        vocab_size,
        context=32,
        embedding=24,
        hidden=48
    ):
        self.vocab_size = vocab_size
        self.context = context
        self.embedding = embedding
        self.hidden = hidden

        rng = np.random.default_rng(42)
        d = embedding

        def weight(a, b):
            return (
                rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
            ).astype(np.float32)

        self.params = {
            "E": (rng.standard_normal((vocab_size, d)) * 0.08).astype(np.float32),
            "P": (rng.standard_normal((context, d)) * 0.02).astype(np.float32),

            "Wq": weight(d, d),
            "Wk": weight(d, d),
            "Wv": weight(d, d),
            "Wo": weight(d, d),

            "W1": weight(d, hidden),
            "b1": np.zeros(hidden, dtype=np.float32),

            "W2": weight(hidden, d),
            "b2": np.zeros(d, dtype=np.float32),

            "Wout": weight(d, vocab_size),
            "bout": np.zeros(vocab_size, dtype=np.float32),
        }

        self.m = {
            k: np.zeros_like(v) for k, v in self.params.items()
        }

        self.v = {
            k: np.zeros_like(v) for k, v in self.params.items()
        }

        self.step = 0

        self.mask = np.triu(
            np.ones((context, context), dtype=bool),
            k=1
        )

    def forward(self, tokens):
        p = self.params

        B, T = tokens.shape
        d = self.embedding

        X = p["E"][tokens] + p["P"][:T]

        Q = X @ p["Wq"]
        K = X @ p["Wk"]
        V = X @ p["Wv"]

        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d)

        scores = np.where(
            self.mask[:T, :T],
            -1e9,
            scores
        )

        scores -= scores.max(axis=-1, keepdims=True)

        A = np.exp(scores)
        A /= A.sum(axis=-1, keepdims=True)

        C = A @ V
        R = X + C @ p["Wo"]

        Z = R @ p["W1"] + p["b1"]
        G = np.maximum(Z, 0)

        H = G @ p["W2"] + p["b2"]
        F = R + H

        logits = F @ p["Wout"] + p["bout"]
        logits -= logits.max(axis=-1, keepdims=True)

        exp = np.exp(logits)
        probs = exp / exp.sum(axis=-1, keepdims=True)

        cache = (tokens, X, Q, K, V, A, C, R, Z, G, F)

        return probs, cache

    def train_step(self, x, y, lr=0.0005):
        p = self.params

        probs, cache = self.forward(x)

        (
            tokens, X, Q, K, V, A,
            C, R, Z, G, F
        ) = cache

        B, T = y.shape
        count = B * T

        selected = probs[
            np.arange(B)[:, None],
            np.arange(T)[None, :],
            y
        ]

        loss = -np.log(selected + 1e-9).mean()

        dz = probs.copy()
        dz[
            np.arange(B)[:, None],
            np.arange(T)[None, :],
            y
        ] -= 1

        dz /= count

        grads = {}

        grads["Wout"] = (
            F.reshape(-1, self.embedding).T
            @ dz.reshape(-1, self.vocab_size)
        )

        grads["bout"] = dz.sum(axis=(0, 1))

        dF = dz @ p["Wout"].T

        dG = dF @ p["W2"].T

        grads["W2"] = (
            G.reshape(-1, self.hidden).T
            @ dF.reshape(-1, self.embedding)
        )

        grads["b2"] = dF.sum(axis=(0, 1))

        dZ = dG * (Z > 0)

        grads["W1"] = (
            R.reshape(-1, self.embedding).T
            @ dZ.reshape(-1, self.hidden)
        )

        grads["b1"] = dZ.sum(axis=(0, 1))

        dR = dF + dZ @ p["W1"].T

        dC = dR @ p["Wo"].T

        grads["Wo"] = (
            C.reshape(-1, self.embedding).T
            @ dR.reshape(-1, self.embedding)
        )

        dA = dC @ V.transpose(0, 2, 1)
        dV = A.transpose(0, 2, 1) @ dC

        dScores = A * (
            dA - (dA * A).sum(axis=-1, keepdims=True)
        )

        dQ = dScores @ K / np.sqrt(self.embedding)

        dK = (
            dScores.transpose(0, 2, 1)
            @ Q / np.sqrt(self.embedding)
        )

        grads["Wq"] = (
            X.reshape(-1, self.embedding).T
            @ dQ.reshape(-1, self.embedding)
        )

        grads["Wk"] = (
            X.reshape(-1, self.embedding).T
            @ dK.reshape(-1, self.embedding)
        )

        grads["Wv"] = (
            X.reshape(-1, self.embedding).T
            @ dV.reshape(-1, self.embedding)
        )

        dX = (
            dR
            + dQ @ p["Wq"].T
            + dK @ p["Wk"].T
            + dV @ p["Wv"].T
        )

        grads["E"] = np.zeros_like(p["E"])

        np.add.at(
            grads["E"],
            tokens.reshape(-1),
            dX.reshape(-1, self.embedding)
        )

        grads["P"] = np.zeros_like(p["P"])
        grads["P"][:T] = dX.sum(axis=0)

        # Adam
        self.step += 1

        beta1 = 0.9
        beta2 = 0.999

        for key in p:
            g = np.clip(grads[key], -1.0, 1.0)

            self.m[key] = beta1 * self.m[key] + (1 - beta1) * g
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (g * g)

            m_hat = self.m[key] / (1 - beta1 ** self.step)
            v_hat = self.v[key] / (1 - beta2 ** self.step)

            p[key] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        return float(loss)

    def predict(self, context_ids, temperature=0.8):
        tokens = np.asarray(
            [context_ids[-self.context:]],
            dtype=np.int64
        )

        probs, _ = self.forward(tokens)

        logits = np.log(probs[0, -1] + 1e-9)
        logits /= max(temperature, 0.05)
        logits -= logits.max()

        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()

        return int(
            np.random.choice(self.vocab_size, p=probabilities)
        )

    def save(self, path):
        data = dict(self.params)

        for key in self.params:
            data["adam_m_" + key] = self.m[key]
            data["adam_v_" + key] = self.v[key]

        data["adam_step"] = np.array(self.step, dtype=np.int64)

        np.savez(path, **data)

    def load(self, path):
        with np.load(path, allow_pickle=False) as data:
            for key in self.params:
                self.params[key] = data[key].astype(np.float32)

                mk = "adam_m_" + key
                vk = "adam_v_" + key

                if mk in data and vk in data:
                    self.m[key] = data[mk].astype(np.float32)
                    self.v[key] = data[vk].astype(np.float32)

            if "adam_step" in data:
                self.step = int(data["adam_step"])
