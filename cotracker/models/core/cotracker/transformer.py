import torch
import torch.nn.functional as F
def transform(input):
    #input: T N 2
    T, N, _ = input.shape
    raw_weights = [[] for _ in range(T)]
    for t in range(T):
        for i in range(N):
            raw_weights[t].append([torch.norm(input[t][i] - input[t][j]) for j in range(N)])

    raw_weights = torch.tensor(raw_weights)
    #Do softmax here mb
    weights = F.softmax(raw_weights, dim=1)
    return weights.matmul(input)