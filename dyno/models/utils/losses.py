
import torch
import torch.nn as nn
import torch.nn.functional as F
    
    
class NTXent(nn.Module):
    def __init__(self, temperature=0.1, weight=0.5, schedule='constant', schedule_kws={}):
        super().__init__()
        self.temperature = temperature
        self.original_weight = weight
        
        self.weight = weight
        
        self.schedule = eval(f'{schedule}_schedule')
        self.schedule_kws = schedule_kws
        
    def get_weight(self, step):
        self.weight = self.schedule(step, self.original_weight, **self.schedule_kws)
        
    
    def forward(self, preds, targets, step = 0):
        
        self.get_weight(step)
        
        
        B, D, T = preds.shape
        labels = torch.arange(B).to(preds.device)

        preds = preds.mean(dim=-1)
        targets = targets.mean(dim=-1)

        computed_sims = F.cosine_similarity(preds.unsqueeze(1), targets.unsqueeze(0), dim=-1) / self.temperature
        computed_sims_2 = F.cosine_similarity(targets.unsqueeze(1), preds.unsqueeze(0), dim=-1) / self.temperature


        loss1 = F.cross_entropy(computed_sims, labels)
        loss2 = F.cross_entropy(computed_sims_2, labels)

        return (loss1 + loss2)
    
