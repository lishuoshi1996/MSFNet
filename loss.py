import torch
class gen_smooth_l1_loss(torch.nn.Module):
    def __init__(self):
        super(gen_smooth_l1_loss, self).__init__()
        self.x0 = 0.02
        self.a = 0.5
        self.b = 0.02
        self.c = 0.0002
    def forward(self, x, y):
        diff = torch.abs(x-y)
        loss = torch.where(diff < self.x0,self.a*torch.pow(diff, 2),self.b*diff-self.c)
        return torch.mean(loss)