import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50
from torchvision.transforms import functional as TF
import random

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
    def __init__(self, input_size=2048, hidden_size=512, out_dim=2048, nonlin='tanh', norm_type=None, n_layers = 2):
        super(prediction_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.norm_type = norm_type
        # 入力x_tを隠れ状態h_tに変換するための重み
        self.i2h = nn.Linear(input_size, hidden_size)
        # 前の隠れ状態h_{t-1}を現在の隠れ状態h_tに変換するための重み
        self.h2h = nn.Linear(hidden_size, hidden_size)

        if n_layers == 2:
            self.output_layer = nn.Linear(hidden_size, out_dim)
        elif n_layers == 3:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, out_dim),
            )

        # 活性化関数の選択
        if nonlin == 'tanh':
            self.activation = torch.tanh
        elif nonlin == 'relu':
            self.activation = torch.relu

        # Normalizationの選択
        if norm_type == 'layernorm':
            self.normalization = nn.LayerNorm(hidden_size)
        
    def forward(self, input_vec):
        # 新しい隠れ状態の計算
        combined = self.i2h(input_vec) + self.h2h(self.hidden)
        if self.norm_type is not None:
            combined = self.normalization(combined)
        self.hidden = self.activation(combined)
        x = self.output_layer(self.hidden)
        return x

    def init_hidden(self, device):
        # 隠れ状態の初期化
        self.hidden = torch.zeros(1, self.hidden_size).to(device)

class prediction_LSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, out_dim=2048, norm_type = None, n_layers = 2):
        super(prediction_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.norm_type = norm_type

        # 入力x_tを各ゲートに変換するための重み
        self.i2f = nn.Linear(input_size, hidden_size)  # 忘却ゲート
        self.i2i = nn.Linear(input_size, hidden_size)  # 入力ゲート
        self.i2o = nn.Linear(input_size, hidden_size)  # 出力ゲート
        self.i2c = nn.Linear(input_size, hidden_size)  # セル状態候補
        
        # 前の隠れ状態h_{t-1}を各ゲートに変換するための重み
        self.h2f = nn.Linear(hidden_size, hidden_size)
        self.h2i = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, hidden_size)
        self.h2c = nn.Linear(hidden_size, hidden_size)

        if n_layers == 2:
            self.output_layer = nn.Linear(hidden_size, out_dim)
        elif n_layers == 3:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, out_dim),
            )

        # LayerNormの設定
        if self.norm_type == 'layernorm':
            self.ln_hidden = nn.LayerNorm(hidden_size)
            self.ln_cell = nn.LayerNorm(hidden_size)
        
    def forward(self, input_vec):
        # ゲートの計算
        f_t = torch.sigmoid(self.i2f(input_vec) + self.h2f(self.hidden))
        i_t = torch.sigmoid(self.i2i(input_vec) + self.h2i(self.hidden))
        o_t = torch.sigmoid(self.i2o(input_vec) + self.h2o(self.hidden))
        
        # セル状態の計算
        c_tilde_t = torch.tanh(self.i2c(input_vec) + self.h2c(self.hidden))
        self.cell = f_t * self.cell + i_t * c_tilde_t
        
        # 隠れ状態の更新
        self.hidden = o_t * torch.tanh(self.cell)
        if self.norm_type is not None:
            self.hidden = self.ln_hidden(self.hidden)
            self.cell = self.ln_cell(self.cell)
        x = self.output_layer(self.hidden)
        return x

    def init_hidden(self, device):
        # 隠れ状態とセル状態の初期化
        self.hidden = torch.zeros(1, self.hidden_size).to(device)
        self.cell = torch.zeros(1, self.hidden_size).to(device)

class RnnSiam(nn.Module):
    def __init__(self, backbone=resnet50(), angle=10, rotate_times = 10, rnn_nonlin = 'tanh', use_aug = False, random_rotation = None, asym_loss = False, rnn_lambda = 0):
        super().__init__()
        self.angle = angle
        self.rotate_times = rotate_times
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)
        self.use_aug = use_aug
        self.random_rotation = random_rotation
        self.sym_loss = not asym_loss
        self.rnn_lambda = rnn_lambda
        self.input_size = 3 * 32 * 32

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )

        self.predictor = prediction_MLP()
        self.rnn_predictor = prediction_RNN(out_dim = self.input_size)
    
    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        
        # 角度の決定
        if self.random_rotation == 'step_random':
            rotate_angle = random.randint(-self.angle, self.angle)
        else:
            rotate_angle = self.angle

        if self.use_aug:
            z1, z2 = f(x1), f(x2)
            xt1 = x1.clone()
            xt2 = x2.clone()
        else:
            z1, z2 = f(x1), f(x1)
            xt1 = x1.clone()
            xt2 = x1.clone()

        self.rnn_predictor.init_hidden(device = x1.device)

        # 回転なしで1回目
        p1, p2 = h(z1), h(z2)
        if self.sym_loss:
            total_loss = D(p1, z2) / 2 + D(p2, z1) / 2
        else:
            total_loss = D(p1, z2) 

        for i in range(self.rotate_times):
            if self.random_rotation == 'time_random':
                rotate_angle = random.randint(-self.angle, self.angle)
            xt1 = TF.rotate(xt1, rotate_angle)
            xt2 = TF.rotate(xt2, rotate_angle)
            zt1, zt2 = f(xt1), f(xt2)
            p1, p2 = h(z1), h(z2)
            if self.sym_loss:
                total_loss += D(p1, zt2) / 2 + D(p2, zt1) / 2
            else:
                total_loss += D(p1, zt2) 
            
            criterion = nn.MSELoss()
            img_predict_1 = self.rnn_predictor(p1)
            img_predict_2 = self.rnn_predictor(p2)
            if self.sym_loss:
                total_loss += self.rnn_lambda * (criterion(img_predict_1, xt2.reshape(-1, self.input_size)) / 2 + criterion(img_predict_2, xt1.reshape(-1, self.input_size)) / 2)
            else:
                total_loss += self.rnn_lambda * criterion(img_predict_2, xt1.reshape(-1, self.input_size)) 

        return {'loss': total_loss / (self.rotate_times+1)}

if __name__ == "__main__":
    model = RnnSiam()
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






