from model import SinusoidalTimeEmbeddings

def test_sinusoidal_time_embeddings():
    import torch
    C = 3
    B = 2
    time_embedder = SinusoidalTimeEmbeddings(C = C)
    t = torch.arange(B)
    t = t.reshape((B,1))
    emb = time_embedder(t)
    assert emb.shape == (B, C)
