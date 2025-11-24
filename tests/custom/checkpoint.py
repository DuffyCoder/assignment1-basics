import os
from typing import BinaryIO, IO
import torch

class Checkpoint:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        self.model = model
        self.optimizer = optimizer
        self.iteration = 0
    
    def save(
        self,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes],
        ):
        torch.save(
            {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iteration': iteration,
            },
            out,
        )
    
    def load(
        self,
        src: str | os.PathLike | BinaryIO | IO[bytes],
    ):
        checkpoint = torch.load(src)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iteration = checkpoint['iteration']
        return self.iteration