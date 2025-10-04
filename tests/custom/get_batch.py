import torch
import numpy.typing as npt

class GetBatch():
    def __init__(
        self,
        dataset: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: str,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        
    def __call__(self):
        indices = torch.randint(0, len(self.dataset) - self.context_length, (self.batch_size,))
        x = torch.zeros((self.batch_size, self.context_length), 
                        device=self.device, dtype=torch.long)
        y = torch.zeros((self.batch_size, self.context_length), 
                        device=self.device, dtype=torch.long)
        for i, start_idx in enumerate(indices):
            x[i] = torch.from_numpy(self.dataset[start_idx:start_idx + self.context_length])
            y[i] = torch.from_numpy(self.dataset[start_idx + 1:start_idx + self.context_length + 1])
        return x, y