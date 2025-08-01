import torch

def compute_L(H):
    ne = H.shape[1]
    nv = H.shape[0]
    W = torch.ones(ne, dtype=torch.float32)
    H = torch.tensor(H, dtype=torch.float32)

    DV = torch.sum(H * W, dim=1)
    DE = torch.sum(H, dim=0)
    DV_sqrt_inv = torch.diag(torch.pow(DV, -0.5))  # NOR
    DE_inv = torch.diag(torch.pow(DE, -1))
    W = torch.diag(W)
    term = DV_sqrt_inv @ H @ W @ DE_inv @ H.t() @ DV_sqrt_inv  # hypergraph filter
    H = term
    I = torch.eye(nv)
    L = I - term
    L_np = L.detach().numpy()
    H_np = H.detach().numpy()

    return L_np, H_np