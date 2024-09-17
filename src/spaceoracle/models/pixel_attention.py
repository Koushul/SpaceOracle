from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pysal.model.spreg import OLS
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch    
import functools
from torch.nn.utils.parametrizations import weight_norm
from torch.distributions import Normal, Gamma

device = torch.device(
    "mps" if torch.backends.mps.is_available() 
    else "cuda" if torch.cuda.is_available() 
    else "cpu"
)


class _cluster_routing(nn.Module):

    def __init__(self, num_clusters, pool_emb, num_experts, dropout_rate=0.1):
        super(_cluster_routing, self).__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.cluster_emb = nn.Embedding(num_clusters, pool_emb)
        self.fc = nn.Linear(pool_emb*2, num_experts)

    def forward(self, spatial_f, labels):
        spatial_f = spatial_f.flatten(1)
        emb = self.cluster_emb(labels)
        x = self.fc(torch.cat([spatial_f, emb], dim=1))
        x = self.dropout(x)
        return F.sigmoid(x)

    # def forward_(self, spatial_f, labels):
    #     spatial_f = spatial_f.flatten()
    #     emb = self.cluster_emb(labels)
    #     x = self.fc(torch.cat([spatial_f, emb]))
    #     x = self.dropout(x)
    #     return F.sigmoid(x)
    


class ConditionalConv2D(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=5, dropout_rate=0.1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(ConditionalConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(2, 2))
        pool_emb = torch.mul(*self._avg_pooling.keywords['output_size']) * in_channels
        
        self._routing_fn = _cluster_routing(
            num_clusters=in_channels,
            pool_emb=pool_emb,
            num_experts=num_experts, 
            dropout_rate=dropout_rate
        )

        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))
        
        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    

    def forward(self, inputs, input_labels):
        res = []
        
        assert inputs.shape[0] == input_labels.shape[0]

        pooled_inputs = self._avg_pooling(inputs)
        routing_weights = self._routing_fn(pooled_inputs, input_labels)
        kernels = torch.sum(routing_weights[:, :, None, None, None, None] * self.weight, 1)
        
        for inputx, kernel in zip(inputs, kernels):
            out = self._conv_forward(inputx.unsqueeze(0), kernel)
            res.append(out)


        # for inputx, label in zip(inputs, input_labels):
        #     inputx = inputx.unsqueeze(0)
        #     pooled_inputs = self._avg_pooling(inputx)
        #     routing_weights = self._routing_fn(pooled_inputs, label)
        #     kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)
        #     out = self._conv_forward(inputx, kernels)
        #     res.append(out)
        
        return torch.cat(res, dim=0)
    



class NicheAttentionNetwork(nn.Module):
     
    def __init__(self, n_regulators, in_channels, spatial_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.spatial_dim = spatial_dim
        self.dim = n_regulators+1
        self.conditional_conv = ConditionalConv2D(self.in_channels, self.in_channels, 1)
        self.sigmoid = nn.Sigmoid()

        self.conv_layers = nn.Sequential(
            weight_norm(nn.Conv2d(in_channels, 32, kernel_size=3, padding='same')),
            nn.PReLU(init=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            weight_norm(nn.Conv2d(32, 64, kernel_size=3, padding='same')),
            nn.PReLU(init=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            weight_norm(nn.Conv2d(64, 128, kernel_size=3, padding='same')),
            nn.PReLU(init=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.cluster_emb = nn.Embedding(self.in_channels, 128)

        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.PReLU(init=0.1),
            nn.Linear(64, 32),
            nn.PReLU(init=0.1),
            nn.Linear(32, self.dim)
        )

        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, spatial_maps, cluster_info):
        att = self.sigmoid(self.conditional_conv(spatial_maps, cluster_info))
        out = att * spatial_maps
        out = self.conv_layers(out)
        emb = self.cluster_emb(cluster_info) * self.alpha
        out = out + emb 

        betas = self.mlp(out)
        
        return betas