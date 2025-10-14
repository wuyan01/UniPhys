import torch
class FixedLengthBuffer:
    def __init__(self, length):
        self.buffer = []
        self.length = length

    def add(self, item):
        if len(self.buffer) >= self.length:
            self.buffer.pop(0)  # Remove the oldest item
        self.buffer.append(item)
        # print(f"Added {item.shape} to buffer. Current buffer")

    def get(self):
        return torch.stack(self.buffer)
    
    def clear(self):
        self.buffer.clear()