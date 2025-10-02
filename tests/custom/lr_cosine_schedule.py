import math

class LrCosineSchedule:
    def __init__(self,
                 it: int,
                 max_learning_rate: float,
                 min_learning_rate: float,
                 warmup_iters: int,
                 cosine_cycle_iters: int):
        self.it = it
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters
    
    def __call__(self):
        if self.it < self.warmup_iters:
            lr =  self.max_learning_rate * self.it / self.warmup_iters
        elif self.it < self.cosine_cycle_iters:
            total_cos_iters = self.cosine_cycle_iters - self.warmup_iters
            if total_cos_iters <= 0:
                lr = self.min_learning_rate
            else:
                cosine_phase = (self.it - self.warmup_iters) / total_cos_iters * math.pi
                lr =  self.min_learning_rate + 0.5 * (self.max_learning_rate - self.min_learning_rate) * (1 + math.cos(cosine_phase))
        else:
            lr = self.min_learning_rate
            
        return lr
