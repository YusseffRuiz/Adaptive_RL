import numpy as np
import torch


class MeanStd(torch.nn.Module):
    def __init__(self, mean=0, std=1, clip=None, shape=None):
        super().__init__()
        self.mean = mean
        self.std = std
        self.clip = clip
        self.count = 0
        self.new_sum = 0
        self.new_sum_sq = 0
        self.new_count = 0
        self.eps = 1e-2
        if shape:
            self.initialize(shape)

    def initialize(self, shape):
        if isinstance(self.mean, (int, float)):
            self.mean = np.full(shape, self.mean, np.float32)
        else:
            self.mean = np.array(self.mean, np.float32)
        if isinstance(self.std, (int, float)):
            self.std = np.full(shape, self.std, np.float32)
        else:
            self.std = np.array(self.std, np.float32)
        self.mean_sq = np.square(self.mean)
        self._mean = torch.nn.Parameter(torch.as_tensor(
            self.mean, dtype=torch.float32), requires_grad=False)
        self._std = torch.nn.Parameter(torch.as_tensor(
            self.std, dtype=torch.float32), requires_grad=False)

    def forward(self, val):
        with torch.no_grad():
            val = (val - self._mean) / self._std
            if self.clip is not None:
                val = torch.clamp(val, -self.clip, self.clip)
        return val

    def unnormalize(self, val):
        return val * self._std + self._mean

    def record(self, values):
        for val in values:
            self.new_sum += val
            self.new_sum_sq += np.square(val)
            self.new_count += 1

    def update(self):
        new_count = self.count + self.new_count
        new_mean = self.new_sum / self.new_count
        new_mean_sq = self.new_sum_sq / self.new_count
        w_old = self.count / new_count
        w_new = self.new_count / new_count
        self.mean = w_old * self.mean + w_new * new_mean
        self.mean_sq = w_old * self.mean_sq + w_new * new_mean_sq
        self.std = self._compute_std(self.mean, self.mean_sq)
        self.count = new_count
        self.new_count = 0
        self.new_sum = 0
        self.new_sum_sq = 0
        self._update(self.mean.astype(np.float32), self.std.astype(np.float32))

    def _compute_std(self, mean, mean_sq):
        var = mean_sq - np.square(mean)
        var = np.maximum(var, 0)
        std = np.sqrt(var)
        std = np.maximum(std, self.eps)
        return std

    def _update(self, mean, std):
        self._mean.data.copy_(torch.as_tensor(self.mean, dtype=torch.float32))
        self._std.data.copy_(torch.as_tensor(self.std, dtype=torch.float32))

class Return(torch.nn.Module):
    def __init__(self, discount_factor):
        super().__init__()
        assert 0 <= discount_factor < 1
        self.coefficient = 1 / (1 - discount_factor)
        self.min_reward = np.float32(-1)
        self.max_reward = np.float32(1)
        self._low = torch.nn.Parameter(torch.as_tensor(
            self.coefficient * self.min_reward, dtype=torch.float32),
            requires_grad=False)
        self._high = torch.nn.Parameter(torch.as_tensor(
            self.coefficient * self.max_reward, dtype=torch.float32),
            requires_grad=False)

    def forward(self, val):
        val = torch.sigmoid(val)
        return self._low + val * (self._high - self._low)

    def record(self, values):
        for val in values:
            if val < self.min_reward:
                self.min_reward = np.float32(val)
            elif val > self.max_reward:
                self.max_reward = np.float32(val)

    def update(self):
        self._update(self.min_reward, self.max_reward)

    def _update(self, min_reward, max_reward):
        self._low.data.copy_(torch.as_tensor(
            self.coefficient * min_reward, dtype=torch.float32))
        self._high.data.copy_(torch.as_tensor(
            self.coefficient * max_reward, dtype=torch.float32))