import torch
import torch.nn as nn
from vnn_net.layers_equi import *


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class VNN_DGCNN(nn.Module):
    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20):
        super(VNN_DGCNN, self).__init__()
        self.c_dim = c_dim
        self.k = k
        
        self.conv_0 = VNLinearLeakyReLU(2, hidden_dim)
        self.conv_1 = VNLinearLeakyReLU(hidden_dim*2, hidden_dim)
        self.conv_2 = VNLinearLeakyReLU(hidden_dim*2, hidden_dim)
        self.conv_3 = VNLinearLeakyReLU(hidden_dim*2, hidden_dim)
        
        self.conv_c = VNLinearLeakyReLU(hidden_dim*4, c_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1).transpose(2, 3)
        
        x = get_graph_mean(x, k=self.k)
        x_0 = self.conv_0(x)
        
        x = get_graph_mean(x_0, k=self.k)
        x_1 = self.conv_1(x)
        
        x = get_graph_mean(x_1, k=self.k)
        x_2 = self.conv_2(x)
        
        x = get_graph_mean(x_2, k=self.k)
        x_3 = self.conv_3(x)
        
        x = torch.cat((x_0, x_1, x_2, x_3), dim=1)
        x = self.conv_c(x)
        x = x.mean(dim=-1, keepdim=False)
        
        return x

class MY_VNN_SimplePointnet(nn.Module):
    ''' DGCNN-based VNN encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None):
        super().__init__()
        self.c_dim = c_dim
        self.k = k
        self.meta_output = meta_output
        
        self.fc_pos = VNLinear(1, hidden_dim//2)
        self.fc_0 = VNLinear(hidden_dim//2, hidden_dim//2)
        self.fc_1 = VNLinear(hidden_dim//2, c_dim)
        self.actvn_0 = VNLeakyReLU(hidden_dim//2, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_1 = VNLeakyReLU(hidden_dim//2, negative_slope=0.0, share_nonlinearity=False)
        
 
        self.fc_pos_ = VNLinear(1, hidden_dim//2)
        self.fc_0_ = VNLinear(hidden_dim//2, hidden_dim//2)
        self.fc_1_ = VNLinear(hidden_dim//2, c_dim)
        self.actvn_0_ = VNLeakyReLU(hidden_dim//2, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_1_ = VNLeakyReLU(hidden_dim//2, negative_slope=0.0, share_nonlinearity=False)
        

        self.pool = meanpool
        
    def forward(self, p):
        # p is B,N,6   32 128 6
        batch_size = p.size(0)
        '''
        p_trans = p.unsqueeze(1).transpose(2, 3)
        
        #net = get_graph_feature(p_trans, k=self.k)
        #net = self.conv_pos(net)
        #net = net.mean(dim=-1, keepdim=False)
        #net = torch.cat([net, p_trans], dim=1)
        
        net = p_trans
        aggr = p_trans.mean(dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, aggr], dim=1)
        '''
        n = p[:,:,3:]
        p = p[:,:,:3]
        p = p.unsqueeze(1).transpose(2, 3) # B,1,3,N
        n = n.unsqueeze(1).transpose(2, 3) # B,1,3,N
        
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn_0(net))
        net = self.fc_1(self.actvn_1(net))
        net = self.pool(net, dim=-1) # B, c_dim, 3

        net_ = self.fc_pos_(n)
        net_ = self.fc_0_(self.actvn_0_(net_))
        net_ = self.fc_1_(self.actvn_1_(net_))
        net_ = self.pool(net_, dim=-1) #

        net = torch.cat([net,net_],axis=1)

        return net.view(batch_size,-1)


class gyj_VNN_SimplePointnet(nn.Module):
    ''' DGCNN-based VNN encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None, feature_transform=True):
        super().__init__()
        self.c_dim = c_dim
        self.k = k
        self.meta_output = meta_output

        self.fc_pos = VNLinear(1, hidden_dim // 2)
        self.fc_0 = VNLinear(hidden_dim // 2, hidden_dim // 2)
        self.fc_1 = VNLinear(hidden_dim // 2, c_dim)
        self.actvn_0 = VNLeakyReLU(hidden_dim // 2, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_1 = VNLeakyReLU(hidden_dim // 2, negative_slope=0.0, share_nonlinearity=False)

        self.fc_pos_ = VNLinear(1, hidden_dim // 2)
        self.fc_0_ = VNLinear(hidden_dim // 2, hidden_dim // 2)
        self.fc_1_ = VNLinear(hidden_dim // 2, hidden_dim // 2)
        self.fc_2_ = VNLinear(hidden_dim // 2, hidden_dim // 2)
        self.fc_3_ = VNLinear(hidden_dim // 2, c_dim)
        self.actvn_0_ = VNLeakyReLU(hidden_dim // 2, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_1_ = VNLeakyReLU(hidden_dim // 2, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_2_ = VNLeakyReLU(hidden_dim // 2, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_3_ = VNLeakyReLU(hidden_dim // 2, negative_slope=0.0, share_nonlinearity=False)

        self.pool = meanpool
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(c_dim=c_dim, dim=dim, d=c_dim, hidden_dim=hidden_dim, meta_output=meta_output)

    def forward(self, p):
        # p is B,N,6   32 128 6
        batch_size = p.size(0)
        '''
        p_trans = p.unsqueeze(1).transpose(2, 3)

        #net = get_graph_feature(p_trans, k=self.k)
        #net = self.conv_pos(net)
        #net = net.mean(dim=-1, keepdim=False)
        #net = torch.cat([net, p_trans], dim=1)

        net = p_trans
        aggr = p_trans.mean(dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, aggr], dim=1)
        '''
        n = p[:, :, 3:]
        p = p[:, :, :3]
        p = p.unsqueeze(1).transpose(2, 3)  # B,1,3,N
        n = n.unsqueeze(1).transpose(2, 3)  # B,1,3,N

        net = self.fc_pos(p)
        net = self.fc_0(self.actvn_0(net))
        net = self.fc_1(self.actvn_1(net))
        net = self.pool(net, dim=-1)  # B, c_dim, 3

        net_ = self.fc_pos_(n)
        net_ = self.fc_0_(self.actvn_0_(net_))
        net_ = self.fc_1_(self.actvn_1_(net_))
        N = net_.size(3)
        # if self.feature_transform:
        #     x_global = self.fstn(net_).unsqueeze(-1).repeat(1, 1, 1, N)
        #     net_ = torch.cat([net_, x_global], axis=1)
        net_ = self.fc_2_(self.actvn_2_(net_))
        net_ = self.fc_3_(self.actvn_3_(net_))
        net_ = self.pool(net_, dim=-1)  #

        net = torch.cat([net, net_], axis=1)

        return net.view(batch_size, -1)


class MY_VNN_SimplePointnet_onlyp(nn.Module):
    ''' DGCNN-based VNN encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=64, k=20, meta_output=None):
        super().__init__()
        self.c_dim = c_dim 
        self.k = k
        self.meta_output = meta_output
        
        self.fc_pos = VNLinear(1, hidden_dim)
        self.fc_0 = VNLinear(hidden_dim, hidden_dim)
        self.fc_1 = VNLinear(hidden_dim, c_dim)
        #self.fc_c = VNLinear(hidden_dim, c_dim)
        
        
        self.actvn_0 = VNLeakyReLU(hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_1 = VNLeakyReLU(hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        
        self.pool = meanpool
        
    def forward(self, p): # [1, 1024, 3]
        batch_size = p.size(1) # 1024
        '''
        p_trans = p.unsqueeze(1).transpose(2, 3)
        
        #net = get_graph_feature(p_trans, k=self.k)
        #net = self.conv_pos(net)
        #net = net.mean(dim=-1, keepdim=False)
        #net = torch.cat([net, p_trans], dim=1)
        
        net = p_trans
        aggr = p_trans.mean(dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, aggr], dim=1)
        '''
        p = p.unsqueeze(1).transpose(2, 3) # B,1,3,1024
        #mean = get_graph_mean(p, k=self.k)
        #mean = p_trans.mean(dim=-1, keepdim=True).expand(p_trans.size())
        
        net = self.fc_pos(p)
        
        net = self.fc_0(self.actvn_0(net))
        net = self.fc_1(self.actvn_1(net))
       
        return net.reshape(-1,batch_size)



class VNN_SimplePointnet(nn.Module):
    ''' DGCNN-based VNN encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None):
        super().__init__()
        self.c_dim = c_dim
        self.k = k
        self.meta_output = meta_output
        
        self.conv_pos = VNLinearLeakyReLU(3, 64, negative_slope=0.0, share_nonlinearity=False, use_batchnorm=False)
        self.fc_pos = VNLinear(64, 2*hidden_dim)
        self.fc_0 = VNLinear(2*hidden_dim, hidden_dim)
        self.fc_1 = VNLinear(2*hidden_dim, hidden_dim)
        self.fc_2 = VNLinear(2*hidden_dim, hidden_dim)
        self.fc_3 = VNLinear(2*hidden_dim, hidden_dim)
        self.fc_c = VNLinear(hidden_dim, c_dim)
        
        
        self.actvn_0 = VNLeakyReLU(2*hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_1 = VNLeakyReLU(2*hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_2 = VNLeakyReLU(2*hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_3 = VNLeakyReLU(2*hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_c = VNLeakyReLU(hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        
        self.pool = meanpool
        
        if meta_output == 'invariant_latent':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
        elif meta_output == 'invariant_latent_linear':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
            self.vn_inv = VNLinear(c_dim, 3)
        
    def forward(self, p):
        batch_size = p.size(0)
        '''
        p_trans = p.unsqueeze(1).transpose(2, 3)
        
        #net = get_graph_feature(p_trans, k=self.k)
        #net = self.conv_pos(net)
        #net = net.mean(dim=-1, keepdim=False)
        #net = torch.cat([net, p_trans], dim=1)
        
        net = p_trans
        aggr = p_trans.mean(dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, aggr], dim=1)
        '''
        p = p.unsqueeze(1).transpose(2, 3)
        #mean = get_graph_mean(p, k=self.k)
        #mean = p_trans.mean(dim=-1, keepdim=True).expand(p_trans.size())
        feat = get_graph_feature_cross(p, k=self.k)
        net = self.conv_pos(feat)
        net = self.pool(net, dim=-1)
        
        net = self.fc_pos(net)
        
        net = self.fc_0(self.actvn_0(net))
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.fc_1(self.actvn_1(net))
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.fc_2(self.actvn_2(net))
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.fc_3(self.actvn_3(net))
        
        net = self.pool(net, dim=-1)

        c = self.fc_c(self.actvn_c(net))
        
        if self.meta_output == 'invariant_latent':
            c_std, z0 = self.std_feature(c)
            return c, c_std
        elif self.meta_output == 'invariant_latent_linear':
            c_std, z0 = self.std_feature(c)
            c_std = self.vn_inv(c_std)
            return c, c_std

        return c

class VNN_ResnetPointnet(nn.Module):
    ''' DGCNN-based VNN encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None):
        super().__init__()
        self.c_dim = c_dim
        self.k = k
        self.meta_output = meta_output

        self.conv_pos = VNLinearLeakyReLU(3, 128, negative_slope=0.0, share_nonlinearity=False, use_batchnorm=False)
        self.fc_pos = VNLinear(128, 2*hidden_dim)
        self.block_0 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = VNLinear(hidden_dim, c_dim)

        self.actvn_c = VNLeakyReLU(hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.pool = meanpool
        
        if meta_output == 'invariant_latent':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
        elif meta_output == 'invariant_latent_linear':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
            self.vn_inv = VNLinear(c_dim, 3)
        elif meta_output == 'equivariant_latent_linear':
            self.vn_inv = VNLinear(c_dim, 3)

    def forward(self, p):
        batch_size = p.size(0)
        p = p.unsqueeze(1).transpose(2, 3)
        #mean = get_graph_mean(p, k=self.k)
        #mean = p_trans.mean(dim=-1, keepdim=True).expand(p_trans.size())
        feat = get_graph_feature_cross(p, k=self.k)
        net = self.conv_pos(feat)
        net = self.pool(net, dim=-1)
        
        net = self.fc_pos(net)
        
        net = self.block_0(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.block_1(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.block_2(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.block_3(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=-1)

        c = self.fc_c(self.actvn_c(net))
        
        if self.meta_output == 'invariant_latent':
            c_std, z0 = self.std_feature(c)
            return c, c_std
        elif self.meta_output == 'invariant_latent_linear':
            c_std, z0 = self.std_feature(c)
            c_std = self.vn_inv(c_std)
            return c, c_std
        elif self.meta_output == 'equivariant_latent_linear':
            c_std = self.vn_inv(c)
            return c, c_std

        return c


class VNN_ResnetPointnet_origin(nn.Module):
    ''' DGCNN-based VNN encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None):
        super().__init__()
        self.c_dim = c_dim
        self.k = k
        self.meta_output = meta_output

        self.conv_pos = VNLinearLeakyReLU(3, 128, negative_slope=0.0, share_nonlinearity=False, use_batchnorm=False)
        self.fc_pos = VNLinear(128, 2*hidden_dim)
        self.block_0 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = VNLinear(hidden_dim, c_dim)

        self.actvn_c = VNLeakyReLU(hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.pool = meanpool
        
        if meta_output == 'invariant_latent':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
        elif meta_output == 'invariant_latent_linear':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
            self.vn_inv = VNLinear(c_dim, 3)
        elif meta_output == 'equivariant_latent_linear':
            self.vn_inv = VNLinear(c_dim, 3)

    def forward(self, p):
        batch_size = p.size(0)
        p = p.unsqueeze(1).transpose(2, 3)
        #mean = get_graph_mean(p, k=self.k)
        #mean = p_trans.mean(dim=-1, keepdim=True).expand(p_trans.size())
        feat = get_graph_feature_cross(p, k=self.k)
        net = self.conv_pos(feat)
        net = self.pool(net, dim=-1)
        
        net = self.fc_pos(net)
        
        net = self.block_0(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.block_1(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.block_2(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.block_3(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=-1)

        c = self.fc_c(self.actvn_c(net))
        
        if self.meta_output == 'invariant_latent':
            c_std, z0 = self.std_feature(c)
            return c, c_std
        elif self.meta_output == 'invariant_latent_linear':
            c_std, z0 = self.std_feature(c)
            c_std = self.vn_inv(c_std)
            return c, c_std
        elif self.meta_output == 'equivariant_latent_linear':
            c_std = self.vn_inv(c)
            return c, c_std

        return c

class VNN_ResnetPointnet_v2(nn.Module):
    ''' DGCNN-based VNN encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None,mode='cnp'):
        super().__init__()
        self.c_dim = c_dim
        self.k = k
        self.meta_output = meta_output
        self.mode = mode

        self.conv_pos = VNLinearLeakyReLU(3, 128, negative_slope=0.0, share_nonlinearity=False, use_batchnorm=False)
        self.fc_pos = VNLinear(128, 2*hidden_dim)
        self.block_0 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = VNLinear(hidden_dim, c_dim)

        self.actvn_c = VNLeakyReLU(hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.pool = meanpool
        
        if meta_output == 'invariant_latent':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
        elif meta_output == 'invariant_latent_linear':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
            self.vn_inv = VNLinear(c_dim, 3)
        elif meta_output == 'equivariant_latent_linear':
            self.vn_inv = VNLinear(c_dim, 3)

    def forward(self, p):
        batch_size = p.size(0)
        if self.mode == 'train':
            p = p.unsqueeze(1).transpose(2, 3)
            #mean = get_graph_mean(p, k=self.k)
            #mean = p_trans.mean(dim=-1, keepdim=True).expand(p_trans.size())
            feat = get_graph_feature_cross(p, k=self.k)
        elif self.mode == 'cnp':
            feat = p
        net = self.conv_pos(feat)
        net = self.pool(net, dim=-1)
        
        net = self.fc_pos(net)
        
        net = self.block_0(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.block_1(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.block_2(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.block_3(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=-1)

        c = self.fc_c(self.actvn_c(net))
        
        if self.meta_output == 'invariant_latent':
            c_std, z0 = self.std_feature(c)
            return c, c_std
        elif self.meta_output == 'invariant_latent_linear':
            c_std, z0 = self.std_feature(c)
            c_std = self.vn_inv(c_std)
            return c, c_std
        elif self.meta_output == 'equivariant_latent_linear':
            c_std = self.vn_inv(c)
            return c, c_std

        return c



class STNkd(nn.Module):
    def __init__(self, c_dim=128, dim=3, d=64, hidden_dim=128, k=20, meta_output=None, global_feat=True,
                     feature_transform=False, pooling='mean'):
        super(STNkd, self).__init__()

        self.conv1 = VNLinearLeakyReLU(d, 64 // 3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 128 // 3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128 // 3, 1024 // 3, dim=4, negative_slope=0.0)

        self.fc1 = VNLinearLeakyReLU(1024 // 3, 512 // 3, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(512 // 3, 256 // 3, dim=3, negative_slope=0.0)

        if pooling == 'max':
            self.pool = VNMaxPool(1024 // 3)
        elif pooling == 'mean':
            self.pool = mean_pool

        self.fc3 = VNLinear(256 // 3, d)
        self.d = d

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
class PointNetEncoder(nn.Module):
    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None, global_feat=True, feature_transform=True, pooling = 'mean'):
        super(PointNetEncoder, self).__init__()
        self.n_knn = 6

        self.conv_pos = VNLinearLeakyReLU(3, 64 // 3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64 // 3, 64 // 3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64 // 3 * 2, 128 // 3, dim=4, negative_slope=0.0)

        self.conv3 = VNLinear(128 // 3, 256 // 3)
        self.bn3 = VNBatchNorm(256 // 3, dim=4)

        self.std_feature = VNStdFeature(256 // 3 * 2, dim=4, normalize_frame=False, negative_slope=0.0)

        if pooling == 'max':
            self.pool = VNMaxPool(64//3)
        elif pooling == 'mean':
            self.pool = mean_pool

        self.global_feat = global_feat
        self.feature_transform = feature_transform

        if self.feature_transform:
            self.fstn = STNkd(c_dim=c_dim,dim=dim, d=64 // 3,hidden_dim=hidden_dim,meta_output=meta_output,pooling=pooling)

    def forward(self, x):
        B, D, N = x.size()
        xx = x[:,0:1,:]
        x = torch.cat([x,xx,xx], dim=1)
        x = x.unsqueeze(1)
        feat = get_graph_feature_cross(x, k=self.n_knn)
        feat = feat.transpose(2, 1)
        x = self.conv_pos(feat)
        x = self.pool(x)

        x = self.conv1(x)

        if self.feature_transform:
            x_global = self.fstn(x).unsqueeze(-1).repeat(1, 1, 1, N)
            x = torch.cat((x, x_global), 1)

        pointfeat = x
        x = self.conv2(x)
        x = self.bn3(self.conv3(x))

        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        x = x.view(B, -1, N)

        x = torch.max(x, -1, keepdim=False)[0]

        trans_feat = None
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
