import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50
from torchvision.transforms import functional as TF

def rotate_image(image, angle):
    """Rotate the image by a given angle."""
    return TF.rotate(image, angle)

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception



class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class prediction_RNN(nn.Module):
    def __init__(self, input_size=2048, hidden_size=2048):
        super(prediction_RNN, self).__init__()
        self.hidden_size = hidden_size
        # 入力x_tを隠れ状態h_tに変換するための重み
        self.i2h = nn.Linear(input_size, hidden_size)
        # 前の隠れ状態h_{t-1}を現在の隠れ状態h_tに変換するための重み
        self.h2h = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input_vec):
        # 新しい隠れ状態の計算
        combined = self.i2h(input_vec) + self.h2h(self.hidden)
        self.hidden = torch.tanh(combined)
        return self.hidden

    def init_hidden(self):
        # 隠れ状態の初期化
        self.hidden = torch.zeros(1, self.hidden_size)

class HipoSiam(nn.Module):
    def __init__(self, backbone=resnet50(), angle=10, rotate_times = 10):
        super().__init__()
        self.angle = angle
        self.rotate_times = rotate_times
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()
        self.rnn_predictor = prediction_RNN()
    
    def forward(self, x1, x2):
        total_loss = 0
        f, h = self.encoder, self.predictor
        xt = x1
        z1 = f(x1)
        # 時計回り
        self.rnn_predictor.init_hidden()
        for i in range(self.rotate_times):
            z2 = f(xt)
            z1 = self.rnn_predictor(z1)
            p1 = h(z1)
            L = D(p1, z2)
            xt = rotate_image(xt, self.angle)
            total_loss += L
        return {'loss': L}

if __name__ == "__main__":
    model = HipoSiam()
    x1 = torch.randn((2, 3, 224, 224))
    x2 = torch.randn_like(x1)

    model.forward(x1, x2).backward()
    print("forward backwork check")

    z1 = torch.randn((200, 2560))
    z2 = torch.randn_like(z1)
    import time
    tic = time.time()
    print(D(z1, z2, version='original'))
    toc = time.time()
    print(toc - tic)
    tic = time.time()
    print(D(z1, z2, version='simplified'))
    toc = time.time()
    print(toc - tic)

# Output:
# tensor(-0.0010)
# 0.005159854888916016
# tensor(-0.0010)
# 0.0014872550964355469












