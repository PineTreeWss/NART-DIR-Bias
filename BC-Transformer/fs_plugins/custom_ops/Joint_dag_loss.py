##########################################################################
# Copyright (C) 2022 COAI @ Tsinghua University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#         http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

#This code implements the hybrid-GLAT alogirthm.
#Returns the best Hybrid-Search Path
import os
import math
import sys

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load
from torch.utils.checkpoint import checkpoint
from torch import jit
from typing import Any, Dict, List, Optional, Tuple
####################### Torch Version of Joint-DAG Oerations ####################
@jit.script
def loop_function_noempty_max(last_f: Tensor, links: Tensor, match: Tensor) -> Tensor:
    f_next = torch.max(last_f + links, dim=1)[0] # batch * 1 * prelen
    f_next = f_next.unsqueeze(-1) + match # batch * prelen * 1
    return f_next
@jit.script
def logsumexp_keepdim(x: Tensor, dim: int) -> Tensor:
    # Solving nan issue when x contains -inf
    # See https://github.com/pytorch/pytorch/issues/31829
    m, _ = x.max(dim=dim, keepdim=True)
    mask = m == -float('inf')
    m = m.detach()
    s = (x - m.masked_fill_(mask, 0)).exp_().sum(dim=dim, keepdim=True)
    return s.masked_fill_(mask, 1).log_() + m.masked_fill_(mask, -float('inf'))
#Perform Lookahead Searching
@jit.script
def loop_function_noempty(last_f: Tensor, links: Tensor, match: Tensor) -> Tensor:
    f_next = logsumexp_keepdim(last_f + links, 1) # batch * 1 * prelen
    f_next = f_next.transpose(1, 2) + match # batch * prelen * 1
    return f_next

def __joint_torch_max_loss(match_all,valid_start_idx_mask, n_links, r_links, output_length, target_length):
    #valid_start_match_all = match_all._masked_fill(mask=valid_start_idx_mask,value='-inf')
    assert n_links.shape[1] == n_links.shape[2], "links should be batch_size * prelen * prelen"
    assert r_links.shape[1] == r_links.shape[2], "links should be batch_size * prelen * prelen"
    #Initiate the valid match likehood as the copy of match_all, and select the most possible position for the first token prediction.
    match_all = match_all.transpose(1, 2)
    batch_size, prelen, tarlen = match_all.shape
    valid_pos_mask = torch.ones(size=(match_all.shape[1:]),dtype=bool,device=match_all.device).T.triu().tril(diagonal=prelen-tarlen)
    #print(valid_pos_mask)
    #exit()
    valid_match_likehood = match_all.clone()
    valid_match_likehood = valid_match_likehood.detach()
    valid_match_likehood = valid_match_likehood.masked_fill_(mask = ~valid_start_idx_mask.unsqueeze(-2).repeat(1,prelen,1),value=float("-inf")).masked_fill_(mask = ~valid_pos_mask.unsqueeze(0).repeat(batch_size,1,1).transpose(1,2),value=float("-inf"))
    #Valid_pos_mask is ok.
    #find_argmax
    max_pos = valid_match_likehood.max(dim=2).values.max(dim=1).indices
    max_tar = valid_match_likehood.max(dim=1).values.max(dim=1).indices
    #print('glat max pos & tar')
    #print(max_pos)
    #print(max_tar)
    #print('output finished')
    #notice that the indices is useful but the value is detached.
    #Backward Search For all samples
    #Output the loss for the first token probabilities
    B_f_arr = []
    t_prob_arr = []
    f_init = torch.zeros(batch_size, prelen, 1, dtype=match_all.dtype, device=match_all.device).fill_(float("-inf"))
    f_init[:, output_length-1, 0] = match_all[:, output_length-1,target_length-1]
    match_all_chunk = torch.chunk(match_all, tarlen, -1) # k * [batch * prelen * 1]
    #Search Backward, refresh the possibility at the argmax position.
    B_f_arr.append(f_init)
    '''print('links')
    print(r_links[51][52])
    print('searching')
    print(f_init[51])'''
    for k in range(tarlen - 2, -1, -1):
        f_now = loop_function_noempty_max(B_f_arr[-1], r_links, match_all_chunk[k])
        eq_mask = max_tar.eq(k).unsqueeze(1).repeat(1,prelen).unsqueeze(-1)
        pos_array = torch.arange(0,prelen,device=match_all.device).repeat(batch_size,1)
        valid_pos_mask = (max_pos.repeat(prelen,1).T == pos_array).unsqueeze(-1)
        inf_mask = f_now.eq(float('-inf')) | ~valid_pos_mask
        f_now.masked_fill_(mask = eq_mask & inf_mask,value=float('-inf'))
        f_now.masked_scatter_(mask = eq_mask & valid_pos_mask,source=torch.zeros(size=f_now.size(),dtype=f_now.dtype,device=match_all.device))
        B_f_arr.append(f_now)
    #print((torch.cat(B_f_arr, -1)).shape)
    #print(batch_size,prelen,tarlen)
    #print(B_f_arr[-1][1])
    #print(torch.cat(B_f_arr, -1)[1].T[-1])
    #for item in torch.cat(B_f_arr,-1)[1]:
    #    print(item)
    #print(torch.cat(B_f_arr, -1)[1][0])
    b_alllogprob = torch.cat(B_f_arr, -1)[range(batch_size),0,-1]
    #print('b_alllogporb')
    #print(b_alllogprob[51])
    #Forward Search
    F_f_arr = []
    f_init = torch.zeros(batch_size, prelen, 1, dtype=match_all.dtype, device=match_all.device).fill_(float("-inf"))
    f_init[:, 0, 0] = match_all[:, 0, 0]
    F_f_arr.append(f_init)
    match_arr = torch.chunk(match_all, tarlen, -1)
    for i in range(1, tarlen):
        f_now = loop_function_noempty_max(F_f_arr[-1], n_links, match_arr[i])
        eq_mask = max_tar.eq(i).unsqueeze(1).repeat(1,prelen).unsqueeze(-1)
        pos_array = torch.arange(0,prelen,device=match_all.device).repeat(batch_size,1)
        valid_pos_mask = (max_pos.repeat(prelen,1).T == pos_array).unsqueeze(-1)
        inf_mask = f_now.eq(float('-inf')) | ~valid_pos_mask
        f_now.masked_fill_(mask = eq_mask & inf_mask,value=float('-inf'))
        f_now.masked_scatter_(mask = eq_mask & valid_pos_mask,source=torch.zeros(size=f_now.size(),dtype=f_now.dtype,device=match_all.device))
        F_f_arr.append(f_now)
    f_alllogprob = torch.cat(F_f_arr, -1)[range(batch_size), output_length - 1, target_length - 1]
    t_alllogprob = []
    for b in range(batch_size):
        t_alllogprob.append(match_all[b][max_pos[b]][max_tar[b]])
    alllogprob = b_alllogprob + f_alllogprob + torch.tensor(t_alllogprob,device=match_all.device,requires_grad=True)
    return alllogprob,(max_pos,max_tar)

def joint_torch_dag_best_alignment(match_all,valid_start_token_mask, n_links, r_links, output_length, target_length):
    """
    Function to obtain the alignment between prediction and reference
    Input:
        match_all (torch.FloatTensor or torch.HalfTensor):
            Shape: [batch_size, max_target_length, max_output_length]
            match_all[b, i, j] represents -log P(y_i| v_j), the probability of predicting the i-th token in the reference
            based on the j-th vertex.
            (Note: float32 are preferred; float16 may cause precision problem)
        links (torch.FloatTensor or torch.HalfTensor):
            Shape: [batch_size, max_output_length, max_transition_length]
            links[b, i, j] represents the transition probability from the i-th vertex to **the j-th vertex**.
            (Note: this parameter is different from the cuda version)
        output_length (torch.LongTensor):
            Shape: [batch_size]
            output_length should be the graph size, the vertices (index >= graph size) are ignored
        target_length (torch.LongTensor):
            Shape: [batch_size]
            target_length is the reference length, the tokens (index >= target length) are ignored

    Output (torch.LongTensor):
        Shape: [batch_size, max_output_length]
        if output[b, i]>=0, it represents the index of target token aligned with the i-th vertex
        otherwise, output[b, i] = -1, it represents the i-th vertex is not aligned with any target token
    """
    with torch.enable_grad():
        match_all.requires_grad_()
        alllogprob,pos = __joint_torch_max_loss(match_all,valid_start_token_mask, n_links, r_links, output_length, target_length)
        matchgrad = torch.autograd.grad(alllogprob.sum(), [match_all])[0] # batch * talen * prelen
    pathvalue, path = matchgrad.max(dim=1)
    path.masked_fill_(pathvalue < 0.5, -1)
    for i in range(len(pos[0])):
        #i = bsz
        #assigh the target pos to path value
        path[i][pos[0][i]]= pos[1][i]
        '''print(pos[1][i])
        print(path[i])
    exit()'''
    return path,pos
def joint_torch_dag_loss(match_all, n_links, r_links, valid_start_idx_mask, output_length, target_length,inf_num=0):
    """
     Function to calculate the joint-dag loss.
     First, search for the maximum token position as the initial position
     , then perform dp search on left and right to find out the probability of the whole sentence.
    Input:
        match_all (torch.FloatTensor or torch.HalfTensor):
            Shape: [batch_size, max_target_length, max_output_length]
            match_all[b, i, j] represents -log P(y_i| v_j), the probability of predicting the i-th token in the reference
            based on the j-th vertex.
            (Note: float32 are preferred; float16 may cause precision problem)
        links (torch.FloatTensor or torch.HalfTensor):
            Shape: [batch_size, max_output_length, max_transition_length]
            links[b, i, j] represents the transition probability from the i-th vertex to **the j-th vertex**.
            (Note: this parameter is different from the cuda version)
        output_length (torch.LongTensor):
            Shape: [batch_size]
            output_length should be the graph size, the vertices (index >= graph size) are ignored
        target_length (torch.LongTensor):
            Shape: [batch_size]
            target_length is the reference length, the tokens (index >= target length) are ignored

    Output (torch.FloatTensor or torch.HalfTensor):
        Shape: [batch_size]
        the loss of each sample
    """
    #print('inf num = ',inf_num)
    match_all = match_all.transpose(1, 2)
    batch_size, prelen, tarlen = match_all.shape
    #print('batch size',batch_size)
    #print('prelen',prelen)
    #print('tarlen',tarlen)
    assert n_links.shape[1] == n_links.shape[2], "links should be batch_size * prelen * prelen"
    assert r_links.shape[1] == r_links.shape[2], "links should be batch_size * prelen * prelen"
    #Look for the argmax token to initiate the searching
    #To encourage the model start from the middle of the sentence, we mask the <pad> <bos> <eos>, and invalid start searching positions.
    move_tokens = int(inf_num/2)
    valid_pos_mask = torch.ones(size=(match_all.shape[1:]),dtype=bool,device=match_all.device).T.triu(diagonal=move_tokens).tril(diagonal= prelen - tarlen - move_tokens)    
    valid_match_likehood = match_all.clone().detach()
    #print(valid_pos_mask.shape)
    #print(valid_pos_mask.unsqueeze(0).repeat(batch_size,1,1).transpose(1,2)[0][0])
    #exit()
    valid_match_likehood = valid_match_likehood.masked_fill_(mask = ~valid_start_idx_mask.unsqueeze(-2).repeat(1,prelen,1),value=float("-inf")).masked_fill_(mask = ~valid_pos_mask.unsqueeze(0).repeat(batch_size,1,1).transpose(1,2),value=float("-inf"))
    max_pos = valid_match_likehood.max(dim=2).values.max(dim=1).indices
    max_tar = valid_match_likehood.max(dim=1).values.max(dim=1).indices
    #print('max_pos & tar values')
    #print(valid_match_likehood.max(dim=2).values[1])
    #print(valid_match_likehood.max(dim=1).values[1])
    #print('match all')
    #for item in match_all[1]:
    #    print(item)
    #print('valid match all')
    #for item in valid_match_likehood[1]:
    #    print(item)
    '''print('pos & tar')
    print(max_pos[:10])
    print(max_tar[:10])'''
    #exit()
    #notice that the indices is useful but the value is detached.
    #Backward Search For all samples
    #Output the loss for the first token probabilities
    B_f_arr = []
    t_prob_arr = []
    f_init = torch.zeros(batch_size, prelen, 1, dtype=match_all.dtype, device=match_all.device).fill_(float("-inf"))
    f_init[:, output_length-1, 0] = match_all[:, output_length-1,target_length-1]
    match_all_chunk = torch.chunk(match_all, tarlen, -1) # k * [batch * prelen * 1]
    #Search Backward, refresh the possibility at the argmax position.
    B_f_arr.append(f_init)
    '''print('length:', target_length[:10])
    print('f_init')
    print(f_init[:10])'''
    #print('link')
    #print(r_links[:10])
    for k in range(tarlen - 2, -1, -1):
        #print('k=',k)
        f_now = loop_function_noempty(B_f_arr[-1], r_links, match_all_chunk[k])
        eq_mask = max_tar.eq(k).unsqueeze(1).repeat(1,prelen).unsqueeze(-1)
        pos_array = torch.arange(0,prelen,device=match_all.device).repeat(batch_size,1)
        valid_pos_mask = (max_pos.repeat(prelen,1).T == pos_array).unsqueeze(-1)
        '''print('valid pos mask')
        print(valid_pos_mask[:10])'''
        inf_mask = f_now.eq(float('-inf')) | ~valid_pos_mask
        f_now.masked_fill_(mask = eq_mask & inf_mask,value=float('-inf'))
        f_now.masked_scatter_(mask = eq_mask & valid_pos_mask,source=torch.zeros(size=f_now.size(),dtype=f_now.dtype,device=match_all.device))
        '''print('f_now')
        print(f_now[:10])'''
        B_f_arr.append(f_now)
    b_alllogprob = torch.cat(B_f_arr, -1)[range(batch_size), 0 , -1]
    '''print('b_f_arr')
    print(B_f_arr[-1][:10])
    print(b_alllogprob[:10])
    exit()'''
    F_f_arr = []
    f_init = torch.zeros(batch_size, prelen, 1, dtype=match_all.dtype, device=match_all.device).fill_(float("-inf"))
    f_init[:, 0, 0] = match_all[:, 0, 0]
    '''print('length:', target_length[:10])
    print('f_init')
    print(f_init[:10])'''
    F_f_arr.append(f_init)
    match_arr = torch.chunk(match_all, tarlen, -1)
    for i in range(1, tarlen):
        #print('i =',i)
        f_now = loop_function_noempty(F_f_arr[-1], n_links, match_arr[i])
        eq_mask = max_tar.eq(i).unsqueeze(1).repeat(1,prelen).unsqueeze(-1)
        pos_array = torch.arange(0,prelen,device=match_all.device).repeat(batch_size,1)
        valid_pos_mask = (max_pos.repeat(prelen,1).T == pos_array).unsqueeze(-1)
        inf_mask = f_now.eq(float('-inf')) | ~valid_pos_mask
        f_now.masked_fill_(mask = eq_mask & inf_mask,value=float('-inf'))
        f_now.masked_scatter_(mask = eq_mask & valid_pos_mask,source=torch.zeros(size=f_now.size(),dtype=f_now.dtype,device=match_all.device))
        F_f_arr.append(f_now)
        '''print('fnow:')
        print(f_now[:10])'''
    f_alllogprob = torch.cat(F_f_arr, -1)[range(batch_size), output_length - 1, target_length - 1]
    t_alllogprob = []
    for b in range(batch_size):
        t_alllogprob.append(match_all[b][max_pos[b]][max_tar[b]])
    loss_result = b_alllogprob + f_alllogprob + torch.tensor(t_alllogprob,device=match_all.device)
    '''print('backlogprob')
    print(b_alllogprob[:10])
    print('forwardlogprob')
    print(f_alllogprob[:10])
    print('t_prob')
    print(t_alllogprob[:10])'''
    return loss_result