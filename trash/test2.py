import torch
import torch.nn as nn

class TestNN(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        self.register_buffer(
            'bool1',
            torch.tensor(False
            , dtype=torch.bool, device=self.device)
        )

        print(self.bool1)

        self.bool2 = nn.Parameter(
            torch.tensor(False, dtype=torch.bool, device=self.device),
            requires_grad=False
        )

        self.register_buffer(
            'bool1',
            torch.tensor(True
            , dtype=torch.bool, device=self.device)
        )
        
        print(self.bool1)

        #self.bool1 += 5

        if not self.bool1:
            print('Hej1')
        
        if not self.bool2:
            print('Hej2')


if __name__ == '__main__':
    TestNN(device='cuda')