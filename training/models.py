import torch.nn as nn


class GlobalForceNet(nn.Module):
    """Predicts global aerodynamic coefficients (CL, log(CD))."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


class FlowFieldNet(nn.Module):
    """
    Predicts local flow field variables (Ux, Uy, Cp, k, omega) at each spatial
    point given geometry coefficients and flow conditions.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 128),   nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        return self.net(x)
