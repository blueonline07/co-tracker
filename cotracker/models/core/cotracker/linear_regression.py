import numpy as np
import torch

class Model:
    def __init__(self, frames):
        self.frames =  frames
        self.T, self.N, self.m = frames.shape

    def fit(self, limit, epochs = 10000, lr = 1e-1):
        params = []
        time = np.linspace(0, 1, limit)
        ones = np.ones_like(time)
        pair = np.stack([time, ones], axis=-1)

        X = np.tile(pair, (self.N, 1, 1))
        y = self.frames[:limit].transpose(1, 0, 2)

        for i in range(self.N):
            w = np.zeros((self.m, self.m))
            for epoch in range(epochs):
                dw = (1 / self.T) * X[i].T.dot(X[i].dot(w) - y[i])
                w = w - lr * dw
                loss = 1 / (2 * self.T) * (X[i].dot(w) - y[i]).T.dot(X[i].dot(w) - y[i])

            params.append(w)
        return params

    def get_coords(self):
        coords = []
        for t in range(self.T):
            params = self.fit(t + 1)
            pred_coords = []
            for i in range(self.N):
                pred_coords.append(np.array([t, 1]).dot(params[i]))
            coords.append(np.array(pred_coords))
        return torch.from_numpy(np.array(coords)[None, :])