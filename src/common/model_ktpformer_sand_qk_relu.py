import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

import time


import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from math import sqrt

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

# 15 _17 spatial 임베딩 차원 h36, eva 에 따라 변경 해주기
# 아이디어 초기에는 저차원의 임베딩 채널, 이후 후반에는 고차원의 임베딩 채널(이때 mlp 앞에도 붙여주기)


# MLP 인풋레이어, 히든 레이어, 아웃풋 레이어 
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., changedim=False, currentdim=0, depth=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x


class LearnableGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    입력 특징과 그래프의 인접 행렬을 사용하여 각 노드의 새로운 특징 표현을 계산함
    """

    def __init__(self, in_features, out_features, adj, bias=True): #
        super(LearnableGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float)) # 2개의 가중치 행렬 W0.shape = (2, 512)
        nn.init.xavier_uniform_(self.W.data, gain=1.414) # 가중치를 Xavier 방식으로 초기화

        self.M = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float)) # 노드 마다 출력 특징을 조정

        # spatial/temporal local topology
        self.adj = adj
        
        # simulated spatial/temporal global topology
        self.adj2 = nn.Parameter(torch.ones_like(adj))        
        nn.init.constant_(self.adj2, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float)) # 편향 b 출력 특징에 더해지는 학습가능한 파라미터
            stdv = 1. / math.sqrt(self.W.size(2)) # 균일 분포로 초기화 
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        # # print(f"Graph conv input shape = {input.shape}")
        h0 = torch.matmul(input, self.W[0]) # 지역 정보 학습
        h1 = torch.matmul(input, self.W[1]) # 글로벌 정보 학습
        
        adj = self.adj.to(input.device) + self.adj2.to(input.device) # 기본 인접 행렬 A와 학습가능한 보정 행렬 A2 더함
  
        adj = (adj.T + adj)/2
        
        E = torch.eye(adj.size(0), dtype=torch.float).to(input.device) # 자기 연결을 표현
        
        output = torch.matmul(adj * E, self.M*h0) + torch.matmul(adj * (1 - E), self.M*h1) # 자기 연결만 포함된 인접행렬 + 비 자기 연결만 포함된 인접행렬, 결과를 더함
        # # print(f"Graph conv output shape = {output.shape}")
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')' 




class KPA(nn.Module):
    """_summary_
    입력 데이터를 그래프 합성곱으로 변환하고, 배치 정규화 및 렐루 적용
    
    """
    def __init__(self, adj, input_dim, output_dim, p_dropout=None): #  input dim=2; output_dim= 32 
        super(KPA, self).__init__()

        self.gconv =  LearnableGraphConv(input_dim, output_dim, adj) # 학습 가능한 그래프 Conv
        self.bn = nn.BatchNorm1d(output_dim) # 배치 정규화
        self.relu = nn.ReLU() # 렐루 

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x): #  x.shape = (b f) n c ;  ([1944, 17, 2]) 
        x = self.gconv(x).transpose(1, 2)   # 그래프 합성공 변환
        x = self.bn(x).transpose(1, 2)      # 배치 정규화
        if self.dropout is not None:        
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        # x.shpae = (b f) n cw([1944, 17, 512])  
        return x



class TPA(nn.Module):
    def __init__(self, adj_temporal, input_dim, output_dim, p_dropout=None):  #  input dim=2; output_dim= 32 
        super(TPA, self).__init__()

        self.gconv =  LearnableGraphConv(input_dim, output_dim, adj_temporal)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class StackedTPA(nn.Module):
    def __init__(self, adj_temporal, input_dim, output_dim, hid_dim, p_dropout): #  #  input dim=2; output_dim= 32 , hid_dim= 32; 
        super(StackedTPA, self).__init__()

        self.gconv1 = TPA(adj_temporal, input_dim, hid_dim, p_dropout)
        self.gconv2 = TPA(adj_temporal, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class Linear_NM(nn.Sequential):
    def __init__(self, input_dim, out_dim):
        super().__init__(
            nn.Linear(input_dim, out_dim),  # Linear layer
            nn.LayerNorm(out_dim)          # LayerNorm applied after Linear
        )
        
class Residual(nn.Module):
    def __init__(self,m):
        super().__init__()
        self.m = m
    
    def forward(self, x):
        return x + self.m(x)
        

# dim= 512
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads # 각 헤드에 입력 피쳐 스플릿
        self.keydim= 16
        attn_ratio = 4
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = self.keydim ** -0.5
        self.vd = int( attn_ratio * self.keydim)  # 64

        # QKV를 위한 독립적인 Linear 레이어 생성; 8개의 키, 쿼리, 밸류 쌍 ; 각각의 Linear를 가짐
        qkvs = []
        dws = []
        dws2 = []
        for i in range(num_heads):
            qkvs.append(Linear_NM(self.head_dim, self.keydim * 2 + self.vd)) # 64 -> 16*2 + 64 차원
            dws.append(nn.Sequential(nn.ReLU(), Linear_NM(self.keydim, self.keydim)))
            dws2.append(nn.Sequential(nn.ReLU(),Linear_NM(self.keydim, self.keydim)))
            
        self.qkvs=nn.ModuleList(qkvs)
        self.dws=nn.ModuleList(dws)
        self.dws2=nn.ModuleList(dws2)
        
        self.proj = nn.Sequential(Linear_NM(self.vd * num_heads, dim),nn.ReLU())

        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj_drop = nn.Dropout(proj_drop)
        
    
    def forward(self, x, vis=False):
        B, N, C = x.shape # (1944, 17, 64). ( (b f)  n cw) -> 64개를 8개로 헤드로 나눠야함
        
        # # print(f"Attention input x shape: {x.shape}") 
        # self.qkv(x) 는 리스트 형태 8개의 (B, N, 3 * self.head_dim)를 가짐
        res = x.clone()
        
        head_outputs =[]
        x_heads = x.chunk(self.num_heads, dim=-1) # 마지막 축을 기준으로 분할, (B, N , hed_dim); (1944, 17, 8) 가 8개있는 리스트; 64개의 feture을 8개로 split
        """_summary_
            self.qkvs의 길이는 self.num_heads.
            각 레이어의 입력 차원은 self.head_dim이고, 출력 차원은 3 * self.head_dim
            i: 현재 반복의 인덱스(0부터 시작).
            head: 현재 i번째 Linear 레이어.
        """
        for i , head in enumerate(self.qkvs):
            feat = x_heads[i]   # 헤드의 i번째 특징 (B, N, head_dim)
             # i > 0일 경우 이전 헤드의 입력과 합산
            if i > 0:
                feat = feat + head_outputs[-1] # 이전 헤드와의 잔차 연결 

            # Query, Key, Value 계산; 각 i번째 Linear 레이어에 feat을 넣고 reshape 이때 Linear를 통해 (B, N, ked_dim * 2 + vd)
            qkv = head(feat)
            q, k, v = torch.split(qkv, [self.keydim, self.keydim, self.vd], dim=-1)
            q = self.dws[i](q)
            k = self.dws2[i](k)
            # 

            # Attention 계산
            attn = (q @ k.transpose(-2, -1)) * self.scale  # Q, K 내적
            attn = attn.softmax(dim=-1)  # Softmax 적용
            attn = self.attn_drop(attn)  # 드롭아웃 적용

            # V에 Attention 적용
            head_out = attn @ v  # B x N x head_dim

            # 해당 헤드 출력 저장
            head_outputs.append(head_out)
        ## # print(f"Attention after head outpus x shape: {head_outputs.shape}") 
        # 모든 헤드의 출력 결합
        x = torch.cat(head_outputs, dim=-1)  # 마지막 차원을 기준으로 연결 -> B x N x C

        # 출력 프로젝션 및 드롭아웃
        x = self.proj(x)
        x = self.proj_drop(x)

        # Residual connection
        x = res + x
        # # print(f"Attention out x shape: {x.shape}") 
        return x 
        




class KPAttention(nn.Module):
    def __init__(self, adj, dim, num_heads=8, drop_path=0, drop_rate=0, norm_layer=nn.LayerNorm, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # 입력데이터를 쿼리, 키, 밸류 변환
        
        self.kpa = KPA(adj, 2, dim, p_dropout=None) # KPA를 통해 키포인트 기반의 그래프 특징 추출

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, 15, dim)) # 키포인트 개수와 입력 차원 dim에 대응하는 위치 임베딩 행렬

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x, vis=False):

        x = self.kpa(x) # 키 포인트 정보가 반영된 그래프 특징 추출

        x += self.Spatial_pos_embed # 각 키포인트의 위치 정보를 더해줌

        x = self.pos_drop(x)

        res = x.clone() # 레즈넷

        x = self.norm1(x)

        B, N, C = x.shape


        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Now x shape (3, B, heads, N, C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) B, heads, N, head_dim 형식
        
        if self.comb==True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb==False:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        

        if self.comb==True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            # print(x.shape)
            x = rearrange(x, 'B H N C -> B N (H C)')
            # print(x.shape)
        elif self.comb==False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)


        x = res + self.drop_path(x)

        return x    




class TPAttention(nn.Module):
    def __init__(self, adj_temporal, num_frame, dim, num_heads=8, drop_path=0, drop_rate=0, norm_layer=nn.LayerNorm, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.tpa = StackedTPA(adj_temporal, dim, dim, dim, p_dropout=None)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x, vis=False):

        x = self.tpa(x)

        x += self.Temporal_pos_embed

        x = self.pos_drop(x)

        res = x.clone() 

        x = self.norm1(x)

        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Now x shape (3, B, heads, N, C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        if self.comb==True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb==False:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        

        if self.comb==True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            # print(x.shape)
            x = rearrange(x, 'B H N C -> B N (H C)')
            # print(x.shape)
        elif self.comb==False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = res + self.drop_path(x)

        return x



class KPA_Block(nn.Module):

    def __init__(self, adj, dim, num_heads, mlp_ratio=4., attention=KPAttention, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0, depth=0, vis=False):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth>0

        # self.norm1 = norm_layer(dim)
        # KPAttention
        self.attn = attention(
            adj, dim, num_heads=num_heads, drop_path=drop_path, drop_rate=drop, norm_layer=nn.LayerNorm, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, comb=comb, vis=vis)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 일부 경로 무작위로 드롭
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # MLP 및 정규화 
        self.norm2 = norm_layer(dim) # C = 32 마지막 차원 정규화 하므로 self.norm2(x).shape = (B, N ,C)
        # print(f" norm 2 dim = {dim}")
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,out_features=dim ,act_layer=act_layer, drop=drop) # 32, 32*4, 32  특징 강화
        
        # 차원 조정 모듈
        """
        중간 이전의 단계에서는 차원을 줄임으로써 전체 모델의 계산량을 줄이고, 메모리 사용량을 절감
        초기에는 단순하고 전역적인 특징을 학습하므로, 고차원의 표현이 꼭 필요하지 않음
        중간 이후 단계에서 차원을 늘려 더 복잡한 관계와 세부 정보를 학습할 수 있음
        모델이 고 해상도의 정보를 표현할 수 있도록 하며, 출력의 세밀함을 보장
        """
        if self.changedim and self.currentdim < self.depth//2:
            self.reduction = nn.Conv1d(dim, dim//2, kernel_size=1)
            # self.reduction = nn.Linear(dim, dim//2)
        elif self.changedim and depth > self.currentdim > self.depth//2:
            self.improve = nn.Conv1d(dim, dim*2, kernel_size=1)
            # self.improve = nn.Linear(dim, dim*2)
        self.vis = vis

    def forward(self, x, vis=False):
        # print(f"KPA Block input x shape: {x.shape}") # torch.Size([1944, 17, 2])
        x = self.attn(x, vis=vis) # x.shape = ((b f) n c)
        # print(f"KPA Block norm2 out shape: {self.norm2(x).shape}") 
        # print(f"KPA Block mlp out shape: {self.mlp(self.norm2(x)).shape}") 
        # 2. Norm2 입력 확인
        try:
            x = x + self.drop_path(self.mlp(self.norm2(x))) # # x.shape = ((b f) n c)
            # print(f"KPA Block after mlp, norm2 x shape: {x.shape}") 
        except Exception as e:
            # print(f"KPA Error during Norm2 or MLP: {e}")
            raise e  # 에러를 다시 발생시켜 트레이스 확인

        # 3. 차원 변경 부분 확인
        if self.changedim and self.currentdim < self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
            # print(f"KPA Block if in x shape: {x.shape}") 
        elif self.changedim and self.depth > self.currentdim > self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
            # print(f"KPA Block elif in x shape: {x.shape}") 
            
        # print(f"KPA Block else in") 
        # print(f"KPA Block out x shape: {x.shape}") 
        return x    # (b f) n c'




class TPA_Block(nn.Module):

    def __init__(self, adj_temporal, num_frame, dim, num_heads, mlp_ratio=4., attention=TPAttention, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0, depth=0, vis=False):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth>0

        # self.norm1 = norm_layer(dim)
        self.attn = attention(
            adj_temporal, num_frame, dim, num_heads=num_heads, drop_path=drop_path, drop_rate=drop, norm_layer=nn.LayerNorm, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, comb=comb, vis=vis)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
        if self.changedim and self.currentdim < self.depth//2:
            self.reduction = nn.Conv1d(dim, dim//2, kernel_size=1)
            # self.reduction = nn.Linear(dim, dim//2)
        elif self.changedim and depth > self.currentdim > self.depth//2:
            self.improve = nn.Conv1d(dim, dim*2, kernel_size=1)
            # self.improve = nn.Linear(dim, dim*2)
        self.vis = vis

    def forward(self, x, vis=False):
        # print(f"TPA Block input x shape: {x.shape}") 
        # x = x + self.drop_path(self.attn(x, vis=vis))
        x = self.attn(x, vis=vis)
        # print(f"TPA Block norm2 out shape: {self.norm2(x).shape}") 
        # print(f"TPA Block mlp out shape: {self.mlp(self.norm2(x)).shape}") 
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        if self.changedim and self.currentdim < self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
            # print(f"TPA Block if in x shape: {x.shape}") 
        elif self.changedim and self.depth > self.currentdim > self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
            # print(f"TPA Block elif in x shape: {x.shape}") 
            
        # print(f"TPA Block else in") 
        # print(f"TPA Block output x shape: {x.shape}") 
        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., attention=Attention, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=True, currentdim=0, depth=0, vis=False):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth>0



        # self.dw0= Residual(Linear_NM(dim,dim))
        # self.ffn0=Residual(Mlp(dim, 2* dim))
        # self.mixer=Residual(attention(
        #     dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, comb=comb, vis=vis))
        # self.dw1= Residual(Linear_NM(dim,dim))
        # self.ffn1= Residual(Mlp(dim, 2* dim))
        
        #self.norm0= norm_layer(dim)
        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, comb=comb, vis=vis)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        #self.mlp0 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    

    def forward(self, x, vis=False):
        # print(f"Block input x shape: {x.shape}") # 1944 17 512 (b f) n cw
        # print(f"Block norm1 out shape: {self.norm1(x).shape}") 
        # print(f"Block norm2 out shape: {self.norm2(x).shape}") 
        # print(f"Block mlp out shape: {self.mlp(self.norm2(x)).shape}") 
        
        # x= self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))
        
        #x=  x+  self.drop_path(self.mlp(self.norm0(x)))
        x = x + self.drop_path(self.attn(self.norm1(x), vis=vis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
    
        return x




"""    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """

class  KTPFormer(nn.Module):
    def __init__(self, adj, adj_temporal, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=512, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = 3     #### output dimension is num_joints * 3

        self.pos_drop = nn.Dropout(p=drop_rate)

        num_drop_path_rate = depth + 1

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_drop_path_rate)]  # stochastic depth decay rule
        self.block_depth = depth

        # 공간적 관계를 모델링, 트랜스 포머 깊이에 따라 블록이 반복적으로 쌓임
        self.STEblocks = nn.ModuleList([
            # Block: Attention Block
             Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+1], norm_layer=norm_layer, changedim=False, currentdim= i+1, depth=depth)
            for i in range(depth)])

        # 수정 ( 블럭을 쌓을때, 서로 다른 임베딩 차원과, 키 차원 할당)

        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+1], norm_layer=norm_layer, comb=False, changedim=False, currentdim=i+1, depth=depth)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        
        # 키포인트 어텐션을 통해 각 관절간의 공간적 관계를 학습
        self.kpattention = KPA_Block(
                adj, dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, attention=KPAttention, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        
        # 시간적 관계 학습
        self.tpattention = TPA_Block(
                adj_temporal, num_frame, dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, attention=TPAttention, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        
        # 최종 출력으로 각 관절의 3차원 좌표를 계산
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )


    def KPA_forward(self, x):
        b, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size
        x = rearrange(x, 'b f n c  -> (b f) n c', ) # ([8, 243, 17, 2]) -> (1944 , 17, 2)

        # print(f"KPA_forward x_shape: {x.shape}")    # (1944 , 17, 2)

        x = self.kpattention(x) # 배치와 프레임을 묶은 후 # 
        # print(f"KPA_forward_attention x_shape: {x.shape}") # [1944, 17, 512])
        
        x = self.Spatial_norm(x) # 마지막 차원 c 정규화
        # print(f"KPA_forward_spatialnorm x_shape: {x.shape}")  # [1944, 17, 512])
        
        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f) # 출력은 b * n , f, c 형태 데이터
        # print(f"KPA_forward_out x_shape: {x.shape}")  # ([8* 17 , 243, 512])
        
        return x

    def TPA_foward(self, x):
        assert len(x.shape) == 3, "shape is equal to 3 ( (b n ), f ,c )"
        b, f, _  = x.shape


        x = self.tpattention(x)

        x = self.Temporal_norm(x) # 마지막 차원 c 정규화
        return x
    
    # 반복적으로 공간적 관계학습, 시간적 관계 학습 
    def ST_foward(self, x):
        assert len(x.shape)==4, "shape is equal to 4"
        # print(f"ST_forward input x_shape: {x.shape}")  # b, f, n, cw ([8, 243, 17, 512])
        b, f, n, cw = x.shape
        for i in range(self.block_depth):
            x = rearrange(x, 'b f n cw -> (b f) n cw')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            # x += self.Spatial_pos_embed
            # x = self.pos_drop(x)
            # if i==7:
            #     x = steblock(x, vis=True)
            x = steblock(x)
            # print(f"ST_forward after steblock x_shape: {x.shape}")  # b f, n, c ; ([8 * 243 , 17, 512 ]) -> ([8 * 243 , 17, 256]) block[i] 에 따라 초기에는 채널의 차원을 낮춤
            x = self.Spatial_norm(x) # 문제가 발생하는 지점
            # print(f"ST_forward after spatial norm x_shape: {x.shape}")  
            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)

            # x += self.Temporal_pos_embed
            # x = self.pos_drop(x)
            # if i==7:
            #     x = tteblock(x, vis=True)
            #     exit()
            x = tteblock(x)
            # print(f"ST_forward after tteblock x_shape: {x.shape}")  
            x = self.Temporal_norm(x)
            # print(f"ST_forward after Temporla norm x shape: {x.shape}")  
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        
        # x = rearrange(x, 'b f n cw -> (b n) f cw', n=n)
        # x = self.weighted_mean(x)
        # x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        # x = x.view(b, f, -1)
        # print(f"ST_ forward output x_shape: {x.shape}")  
        return x

    def forward(self, x):
        b, f, n, c = x.shape # torch.Size([8, 243, 17, 2])

        # print(f"KTPFormer input x_shape: {x.shape}")
        ### now x is [batch_size, channels, receptive frames, joint_num], following image data
        # x shape:(b f n c)
        
        # st = time.time()
        x = self.KPA_forward(x)
        # print(f"KTPFormer after KPA_forward x_shape: {x.shape}")
        # now x shape is (b n) f cw 가중치 학습
        # et = time.time()
        # # print('STE_forward  ', (et-st)*2000)

        # st = time.time()
        x = self.TPA_foward(x)
        # print(f"KTPFormer after TPA_forward x_shape: {x.shape}") # 여기까지 # ([8 * 17, 243, 512])
        # et = time.time()
        # # print('TTE_foward  ', (et-st)*2000)

        # now x shape is (b n) f cw
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n) # ([8, 243, 17, 512])
        # st = time.time()
        x = self.ST_foward(x)
        # print(f"KTPFormer after ST_forward x_shape: {x.shape}") # ([8, 243, 17, 512])
        # et = time.time()
        # # print('ST_foward  ', (et-st)*2000)

        # st = time.time()
        x = self.head(x)
        # et = time.time()
        # # print('head  ', (et-st)*2000)
        # now x shape is (b f (n * 3))

        x = x.view(b, f, n, -1) # x의 최종 출력은 (batch_size, frame, num_joint, 3), 이므로 모든 관절의 3차원 좌표를 포함하는 전체 출력

        return x







