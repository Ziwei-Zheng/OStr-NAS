# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from nasbench2_ops import *


def gen_searchcell_mask_from_arch_str(arch_str):
    nodes = arch_str.split('+') 
    nodes = [node[1:-1].split('|') for node in nodes]
    nodes = [[op_and_input.split('~')  for op_and_input in node] for node in nodes]

    keep_mask = []
    for curr_node_idx in range(len(nodes)):
            for prev_node_idx in range(curr_node_idx+1): 
                _op = [edge[0] for edge in nodes[curr_node_idx] if int(edge[1]) == prev_node_idx]
                assert len(_op) == 1, 'The arch string does not follow the assumption of 1 connection between two nodes.'
                for _op_name in OPS.keys():
                    keep_mask.append(_op[0] == _op_name)
    return keep_mask


def get_model_from_arch_str(arch_str, num_classes, use_bn=True, init_channels=16):
    keep_mask = gen_searchcell_mask_from_arch_str(arch_str)
    net = NAS201Model(arch_str=arch_str, num_classes=num_classes, use_bn=use_bn, keep_mask=keep_mask, stem_ch=init_channels)
    return net


def get_super_model(num_classes, use_bn=True):
    net = NAS201Model(arch_str=arch_str, num_classes=num_classes, use_bn=use_bn)
    return net


class NAS201Model(nn.Module):

    def __init__(self, num_classes, use_bn=True, keep_mask=None, stem_ch=16, layers=3):
        super(NAS201Model, self).__init__()
        self.num_classes=num_classes
        self.use_bn= use_bn

        self.layers = layers

        self.stem = stem(out_channels=stem_ch, use_bn=use_bn)
        self.stack_cell1 = nn.ModuleList()
        self.reduction1 = reduction(in_channels=stem_ch, out_channels=stem_ch*2)
        self.stack_cell2 = nn.ModuleList()
        self.reduction2 = reduction(in_channels=stem_ch*2, out_channels=stem_ch*4)
        self.stack_cell3 = nn.ModuleList()

        for i in range(5):
            self.stack_cell1 += [SearchCell(C=stem_ch, stride=1)]
            self.stack_cell2 += [SearchCell(C=stem_ch*2, stride=1)]
            self.stack_cell3 += [SearchCell(C=stem_ch*4, stride=1)]

        self.top = top(in_dims=stem_ch*4, num_classes=num_classes, use_bn=use_bn)

        self.init_additional_para()


    def forward(self, x):
        x = self.stem(x)

        self.weights = F.softmax(self.alphas, dim=-1)

        for i in range(5):
            x = self.stack_cell1[i](x, self.weights * self.c)
        x = self.reduction1(x)

        for i in range(5):
            x = self.stack_cell2[i](x, self.weights * self.c)
        x = self.reduction2(x)

        for i in range(5):
            x = self.stack_cell3[i](x, self.weights * self.c)

        x = self.top(x)
        return x

    def init_additional_para(self):
        k, num_ops = 6, len(PRIMITIVES)

        self.alphas = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    
        self.weights = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)

        self.c = Variable(torch.ones(k, num_ops).cuda(), requires_grad=True)


def get_arch_str(metrics, is_abs):
    metrics_nz = metrics[:, 1:]
    if is_abs:
        metrics_nz = torch.abs(metrics_nz)
    nodes = []
    idx = 0
    for i in range(3):
        edges = []
        for j in range(i+1):
            edges.append(f'{PRIMITIVES[metrics_nz[idx].argmax()+1]}~{j}')
            idx += 1
        node_str = '|'.join(edges)
        node_str = f'|{node_str}|'
        nodes.append(node_str)
    return '+'.join(nodes)
    

def get_arch_str_from_model(net):
    search_cell = net.stack_cell1[0].options
    keep_mask = net.stack_cell1[0].keep_mask
    num_nodes = net.stack_cell1[0].num_nodes

    nodes = []
    idx = 0
    for curr_node in range(num_nodes -1):
        edges = []
        for prev_node in range(curr_node+1): # n-1 prev nodes
            for _op_name in OPS.keys():
                if keep_mask[idx]:
                    edges.append(f'{_op_name}~{prev_node}')
                idx += 1
        node_str = '|'.join(edges)
        node_str = f'|{node_str}|'
        nodes.append(node_str) 
    arch_str = '+'.join(nodes)
    return arch_str


if __name__ == "__main__":
    arch_str = '|nor_conv_3x3~0|+|none~0|none~1|+|avg_pool_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|'
    
    n = get_model_from_arch_str(arch_str=arch_str, num_classes=10)
    print(n.stack_cell1[0])
    
    arch_str2 = get_arch_str_from_model(n)
    print(arch_str)
    print(arch_str2)
    print(f'Are the two arch strings same? {arch_str == arch_str2}')
