import torch
import torch.nn as nn
import math

class BaseModel(nn.Module):
    def __init__(self, channels=6, hidden=64): 
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(64, hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )

    def forward(self, x):
        # [QUAN TRỌNG CHÍ MẠNG] Thêm .contiguous() để chống sập lõi MKL/cuDNN trên Windows
        x = x.permute(0, 2, 1).contiguous()
        x = self.cnn(x)
        x = x.permute(0, 2, 1).contiguous()
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))

class AdaptiveFuzzyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu_conf = nn.Parameter(torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]))
        self.sigma_conf = nn.Parameter(torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]))
        self.mu_noise = nn.Parameter(torch.tensor([0.2, 0.5, 0.8, 1.1, 1.5]))
        self.sigma_noise = nn.Parameter(torch.tensor([0.3, 0.3, 0.3, 0.3, 0.3]))
        
        init_weights = torch.zeros(5, 5)
        for i, c in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
            for j, n in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
                init_weights[i, j] = c 
                
        def logit(p):
            p = max(min(p, 0.99), 0.01)
            return math.log(p / (1 - p))
            
        init_logits = torch.tensor([logit(x.item()) for x in init_weights.view(-1)])
        self.rule_logits = nn.Parameter(init_logits)
        self.alpha = nn.Parameter(torch.tensor(0.0)) 

    def forward(self, conf, noise):
        conf = conf.view(-1, 1).contiguous()
        noise = noise.view(-1, 1).contiguous()
        
        mu_c = torch.exp(-0.5 * ((conf - self.mu_conf) / (torch.abs(self.sigma_conf) + 1e-4)) ** 2)
        mu_n = torch.exp(-0.5 * ((noise - self.mu_noise) / (torch.abs(self.sigma_noise) + 1e-4)) ** 2)
        
        firing = torch.bmm(mu_c.unsqueeze(2), mu_n.unsqueeze(1)).view(conf.size(0), -1)
        rule_w = torch.sigmoid(self.rule_logits)
        num = torch.sum(firing * rule_w, dim=1, keepdim=True)
        den = torch.sum(firing, dim=1, keepdim=True) + 1e-6
        return num / den

class ProposedModel(nn.Module):
    def __init__(self, channels=6):
        super().__init__()
        self.base = BaseModel(channels)
        self.fuzzy = AdaptiveFuzzyLayer()

    def forward(self, x, noise):
        conf = self.base(x)
        fuzzy_out = self.fuzzy(conf, noise)
        alpha_val = torch.sigmoid(self.fuzzy.alpha)
        return alpha_val * conf + (1.0 - alpha_val) * fuzzy_out