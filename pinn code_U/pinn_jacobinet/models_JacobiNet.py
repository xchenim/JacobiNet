import torch.nn as nn

class jacobinet(nn.Module):
    def __init__(self):
        super(jacobinet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.main(x)
