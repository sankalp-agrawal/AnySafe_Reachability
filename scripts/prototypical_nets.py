import random
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import trange


# -------------------------
# MLP Encoder
# -------------------------
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Euclidean distance
# -------------------------
def euclidean_distance(x, y):
    return torch.norm(x - y, p=2)


# -------------------------
# Compute prototype
# -------------------------
def compute_prototype(features):
    return torch.stack(features).mean(dim=0)


# -------------------------
# Generate episodic loss
# -------------------------
def prototypical_loss(D, K, NC, NS, NQ, encoder, device):
    D_class = defaultdict(list)
    for x, y in D:
        D_class[y].append(x.to(device))

    V = random.sample(range(K), NC)  # Class indices for episode
    prototypes = {}
    query_sets = {}

    for k in V:
        examples = D_class[k]
        support = random.sample(examples, NS)
        remaining = list(set(examples) - set(support))
        query = random.sample(remaining, NQ)

        support_feats = [encoder(x.unsqueeze(0)).squeeze(0) for x in support]
        prototypes[k] = compute_prototype(support_feats)
        query_sets[k] = query

    loss = 0.0
    for k in V:
        for x in query_sets[k]:
            x_feat = encoder(x.unsqueeze(0)).squeeze(0)
            dists = torch.stack(
                [euclidean_distance(x_feat, prototypes[kp]) for kp in V]
            )
            log_prob = -dists[V.index(k)] + torch.logsumexp(-dists, dim=0)
            loss += -log_prob

    return loss / (NC * NQ)


# -------------------------
# Dummy Dataset
# -------------------------
def generate_dummy_dataset(K=5, N=100, input_dim=20):
    D = []
    for k in range(K):
        for _ in range(N):
            x = torch.randn(input_dim) + 5 * k  # shift mean per class
            y = k
            D.append((x, y))
    return D


# -------------------------
# Training loop
# -------------------------
def train_prototypical_net():
    # Hyperparameters
    input_dim = 20
    hidden_dim = 64
    embedding_dim = 32
    K = 5
    NC = 3
    NS = 5
    NQ = 5
    num_episodes = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    D = generate_dummy_dataset(K=K, N=50, input_dim=input_dim)

    # Model
    encoder = MLPEncoder(input_dim, hidden_dim, embedding_dim).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    # Training
    for episode in trange(num_episodes, desc="Training Episodes"):
        encoder.train()
        loss = prototypical_loss(D, K, NC, NS, NQ, encoder, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}: Loss = {loss.item():.4f}")

    return encoder


# Run it
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = train_prototypical_net().to(device)

    D_val = generate_dummy_dataset(K=5, N=10, input_dim=20)
    D_class = defaultdict(list)
    for x, y in D_val:
        D_class[y].append(x.to(device))

    prototypes = {}
    for key in D_class.keys():
        with torch.no_grad():
            support_feats = [encoder(x.unsqueeze(0)).squeeze(0) for x in D_class[key]]
        prototypes[key] = compute_prototype(support_feats)

    import ipdb

    ipdb.set_trace()  # noqa: E402

    x = torch.randn(20) * 2 + 5 * 2  # shift mean per class
    x = x.to(device)
