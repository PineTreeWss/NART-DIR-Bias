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
#from memory_profiler import profile
import math
import re
import logging
from functools import reduce
import numpy as np
from typing import Union, Tuple, Optional
import sys

import torch
from torch import Tensor
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch.autograd import Function
from ..custom_ops import dag_loss, rev_dag_loss,dag_best_alignment, dag_logsoftmax_gather_inplace, torch_dag_loss, torch_dag_best_alignment, torch_dag_logsoftmax_gather_inplace,  reverse_torch_dag_loss, reverse_torch_dag_best_alignment, reverse_torch_dag_logsoftmax_gather_inplace, joint_torch_dag_best_alignment,joint_torch_dag_loss

from .utilities import parse_anneal_argument, get_anneal_value

logger = logging.getLogger(__name__)

########### gpu use tracker ###########
#import inspect
SHOW_MEMORY_USE=False
if SHOW_MEMORY_USE:
    from fairseq.gpu_mem_track import MemTracker
    gpu_tracker = MemTracker()
########################################

@register_criterion("joint_bi_nat_dag_loss")
class JointBiNATDAGLoss(FairseqCriterion):

    def __init__(self, cfg, task):
        super().__init__(task)
        self.cfg = cfg
        assert cfg.label_smoothing == 0, "DAG does not support label smoothing"
        self.glance_strategy = cfg.glance_strategy
        self._glat_p_anneal_params = parse_anneal_argument(cfg.glat_p)

        self.set_update_num(0)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument("--label-smoothing", type=float, default=0, help="DA-Transformer does not use label smoothing for now")
        parser.add_argument("--glat-p", type=str, default="0", help="Glancing probability. 0.5:0.1@200k indicates annealing p from 0.5 to 0.1 in 200k steps.")
        parser.add_argument("--glance-strategy", type=str, default=None, help='Glancing strategy. Possible values: "number-random" or "None" or "CMLM" or "Joint", wher "Joint" indicates thr Joint GLAT training of two graphs.')
        parser.add_argument("--no-force-emit", action="store_true", help="If true, do not fix the position of glance tokens in the second forward pass")

        parser.add_argument("--torch-dag-logsoftmax-gather", action="store_true", help="Use torch implementation for logsoftmax-gather, which supports GPU and CPU device. (Cuda implementation only supports GPU)")
        parser.add_argument("--torch-dag-best-alignment", action="store_true", help="Use torch implementation for dag-best-alignment, which supports GPU and CPU device. (Cuda implementation only supports GPU)")
        parser.add_argument("--torch-dag-loss", action="store_true", help="Use torch implementation for dag-loss, which supports GPU and CPU device. (Cuda implementation only supports GPU)")

    def _compute_loss(self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = utils.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                        nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss_nofactor = loss
        loss = loss * factor

        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor, "ntokens": outputs.shape[0], "loss_nofactor": loss_nofactor}
    def _compute_joint_dag_loss(self, outputs, output_masks, valid_start_mask,targets, target_masks, n_links, r_links,label_smoothing=0.0, name="loss",
                factor=1.0, dir = 'n', matchmask=None, keep_word_mask=None, model=None,glat_keep=None):
        if dir != 'j':
            print('Wrong Link Direction!')
            exit()
        batch_size = outputs.shape[0]
        prelen = outputs.shape[1]
        tarlen = targets.shape[1]
        output_length = output_masks.sum(dim=-1)
        target_length = target_masks.sum(dim=-1)
        if self.cfg.torch_dag_logsoftmax_gather:
            outputs, match_all = torch_dag_logsoftmax_gather_inplace(outputs, targets.unsqueeze(1).expand(-1, prelen, -1))
        else:
            outputs, match_all = dag_logsoftmax_gather_inplace(outputs, targets.unsqueeze(1).expand(-1, prelen, -1))
        match_all = match_all.transpose(1, 2)
        #print(match_all[1])
        if matchmask is not None and not self.cfg.no_force_emit:
            glat_prev_mask = keep_word_mask.unsqueeze(1)
            match_all = match_all.masked_fill(glat_prev_mask, 0) + match_all.masked_fill(~matchmask, float("-inf")).masked_fill(~glat_prev_mask, 0).detach()
        nvalidtokens = output_masks.sum()
        '''print(glat_prev_mask[1])
        print(matchmask[1])
        print(match_all[1])'''
        if self.cfg.torch_dag_loss:
            if dir == 'j':
                if model.args.max_transition_length != -1:
                    n_links = model.restore_valid_n_links(n_links)
                    r_links = model.restore_valid_r_links(r_links)
                valid_start_tokens_mask = ~(targets.eq(model.pad) | targets.eq(model.bos) | targets.eq(model.eos)) & valid_start_mask
                inf_num = int(glat_keep * prelen)
                loss_result = joint_torch_dag_loss(match_all, n_links, r_links, valid_start_tokens_mask, output_length, target_length,inf_num)
        else:
            print("cuda_joint_dag_loss not implemented!")
            assert model.args.max_transition_length != -1, "cuda dag loss does not support max_transition_length=-1. You can use a very large number such as 99999"
            #loss_result = dag_loss(match_all, links, output_length, target_length)
            exit()
        '''print('loss:')
        print(loss_result)
        nan_idx = []
        for i in range(batch_size):
            if (loss_result[i]) == float('-inf'):
                nan_idx.append(i)
        print(batch_size)
        print(nan_idx)
        print(len(nan_idx)/batch_size)
        exit()'''
        invalid_masks = loss_result.isinf().logical_or(loss_result.isnan())
        #print('masked_fill loss')
        loss_result.masked_fill_(invalid_masks, 0)
        invalid_nsentences = invalid_masks.sum().detach()
        #print(loss_result)
        loss = -(loss_result / target_length).mean()
        #print('length normalized loss')
        #print(loss)
        nll_loss = loss.detach()
        nsentences, ntokens = targets.shape[0], targets.ne(self.task.tgt_dict.pad()).sum()

        loss_nofactor = loss
        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss,
                "factor": factor, "ntokens": ntokens, "nvalidtokens": nvalidtokens, "nsentences": nsentences,
                "loss_nofactor": loss_nofactor, "invalid_nsentences": invalid_nsentences}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def set_update_num(self, update_num):
        self.glat_p = get_anneal_value(self._glat_p_anneal_params, update_num)
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # import gc
        # gc.collect()
        if SHOW_MEMORY_USE:
            print(torch.cuda.memory_reserved() / 1024 / 1024, file=sys.stderr, flush=True)
            gpu_tracker.clear_cache()
        # gpu_tracker.track()
        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens = sample["target"]

        if SHOW_MEMORY_USE:
            print(sample["net_input"]["src_tokens"].shape[0], sample["net_input"]["src_tokens"].shape[1], tgt_tokens.shape[1], file=sys.stderr, end=" ")

        if sample.get("update_num", None) is not None: # in training
            self.set_update_num(sample['update_num'])

        prev_output_tokens = model.initialize_output_tokens_by_tokens(src_tokens, tgt_tokens)

        if self.glat_p == 0:
            glat = None
        else:
            glat = {
                "context_p": max(self.glat_p, 0),
                "require_glance_grad": False
            }
        def find_max_positions(model,word_ins_out,tgt_tokens):
            #Input the model output ant the target tokens, find out the argmax target tokens in the valid position.
            #Return: Position (Torch, len = b argmax token id from [0,prelen]) Target (Torch, len = b, target token id from [0,tgt_tokens_num]) 
            batch_size,prelen,tokennum = word_ins_out.shape
            tarlen = tgt_tokens.shape[1]
            nonpad_positions = ~tgt_tokens.eq(model.pad)
            pred_tokens = word_ins_out.argmax(-1)
            if self.cfg.torch_dag_logsoftmax_gather:
                word_ins_out, match = torch_dag_logsoftmax_gather_inplace(word_ins_out, tgt_tokens.unsqueeze(1).expand(-1, prelen, -1))
            else:
                word_ins_out, match = dag_logsoftmax_gather_inplace(word_ins_out, tgt_tokens.unsqueeze(1).expand(-1, prelen, -1))
            valid_pos_mask = torch.ones(size=(match.shape[1:]),dtype=bool,device=match.device).T.triu().tril(diagonal=prelen-tarlen)
            valid_match_likehood = match.clone().detach()
            valid_start_tokens_mask = ~(tgt_tokens.eq(model.pad) | tgt_tokens.eq(model.bos) | tgt_tokens.eq(model.eos))
            valid_match_likehood = valid_match_likehood.masked_fill_(mask = ~valid_start_tokens_mask.unsqueeze(-2).repeat(1,prelen,1),value=float("-inf")).masked_fill_(mask = ~valid_pos_mask.unsqueeze(0).repeat(batch_size,1,1).transpose(1,2),value=float("-inf"))
            #Valid_pos_mask is ok.
            #valid_match_all = match_all.masked_fill_(mask = ~valid_start_idx_mask.unsqueeze(-2).repeat(1,prelen,1),value=float("-inf")).masked_fill_(mask = ~valid_pos_mask.unsqueeze(0).repeat(batch_size,1,1).transpose(1,2),value=float("-inf"))
            #print(match_all)
            #find_argmax
            max_pos = valid_match_likehood.max(dim=2).values.max(dim=1).indices
            max_tar = valid_match_likehood.max(dim=1).values.max(dim=1).indices
            return max_pos,max_tar
        def glat_function(model, word_ins_out, tgt_tokens, prev_output_tokens,  glat,valid_tgt_pos_mask,dir='n',links=None):
            if dir=='j':
                batch_size, prelen, _ = links[0].shape
            else:
                batch_size, prelen, _ = links.shape
            tarlen = tgt_tokens.shape[1]
            nonpad_positions = ~tgt_tokens.eq(model.pad)
            target_length = (nonpad_positions).sum(1)
            output_length = prev_output_tokens.ne(model.pad).sum(1)
            pred_tokens = word_ins_out.argmax(-1)
            if self.cfg.torch_dag_logsoftmax_gather:
                word_ins_out, match = torch_dag_logsoftmax_gather_inplace(word_ins_out, tgt_tokens.unsqueeze(1).expand(-1, prelen, -1))
            else:
                word_ins_out, match = dag_logsoftmax_gather_inplace(word_ins_out, tgt_tokens.unsqueeze(1).expand(-1, prelen, -1))
            match = match.transpose(1, 2)
            shreshold_pos = -1
            if dir=='j':
                #valid_start_token_pos = 
                valid_start_tokens_mask = (~(tgt_tokens.eq(model.pad) | tgt_tokens.eq(model.bos) | tgt_tokens.eq(model.eos))) & valid_tgt_pos_mask
                n_link = model.restore_valid_n_links(links[0])
                r_link = model.restore_valid_r_links(links[1])
                path,shreshold_pos = joint_torch_dag_best_alignment(match,valid_start_tokens_mask,n_link,r_link,output_length,target_length)
            else:
                if self.cfg.torch_dag_best_alignment:
                    if model.args.max_transition_length != -1:
                        if dir == 'n':
                            links = model.restore_valid_n_links(links)
                        elif dir == 'r':
                            links = model.restore_valid_r_links(links)
                    if dir == 'n':
                        path = torch_dag_best_alignment(match, links, output_length, target_length)
                    elif dir == 'r':
                        path = reverse_torch_dag_best_alignment(match, links, output_length, target_length)
                else:
                    assert model.args.max_transition_length != -1, "cuda dag best alignment does not support max_transition_length=-1. You can use a very large number such as 99999"
                    path = dag_best_alignment(match, links, output_length, target_length) # batch * prelen
            predict_align_mask = path >= 0
            matchmask = torch.zeros(batch_size, tarlen + 1, prelen, device=match.device, dtype=torch.bool).scatter_(1, path.unsqueeze(1) + 1, 1)[:, 1:]
            '''print('Outpu Path')
            for i in range(batch_size):
                print(i)
                print(path[i])
                print(shreshold_pos[0][i],shreshold_pos[1][i])'''
                #print(matchmask[i])
            #print(path)
            oracle = tgt_tokens.gather(-1, path.clip(min=0)) # bsz * prelen
            same_num = ((pred_tokens == oracle) & predict_align_mask).sum(1)

            if self.glance_strategy is None:
                keep_prob = ((target_length - same_num) / target_length * glat['context_p']).unsqueeze(-1) * predict_align_mask.float()

            elif self.glance_strategy in ['number-random']:
                prob = torch.randn(oracle.shape, device=tgt_tokens.device, dtype=torch.float)
                prob.masked_fill_(~predict_align_mask, -100)
                glance_nums = ((target_length - same_num) * glat['context_p'] + 0.5).to(torch.long)
                #prob_thresh = prob.topk(glance_nums.max().clip(min=1))[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh = prob.sort(descending=True)[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh.masked_fill_(glance_nums == 0, 100)
                keep_prob = (prob >= prob_thresh.unsqueeze(-1)).to(prob.dtype)
            elif self.glance_strategy == "cmlm":
                if dir == 'j':
                    print('glancing for cmlm not implemented for Joint Loss')
                    exit()
                prob = torch.randn(oracle.shape, device=tgt_tokens.device, dtype=torch.float)
                prob.masked_fill_(~predict_align_mask, -100)
                glance_nums = (target_length * torch.rand_like(target_length, dtype=torch.float) + 0.5).to(torch.long)
                #prob_thresh = prob.topk(glance_nums.max().clip(min=1))[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh = prob.sort(descending=True)[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh.masked_fill_(glance_nums == 0, 100)
                keep_prob = (prob >= prob_thresh.unsqueeze(-1)).to(prob.dtype)
            keep_word_mask = (torch.rand(prev_output_tokens.shape, device=prev_output_tokens.device) < keep_prob).bool()

            glat_prev_output_tokens = prev_output_tokens.masked_fill(keep_word_mask, 0) + oracle.masked_fill(~keep_word_mask, 0)
            glat_tgt_tokens = tgt_tokens

            glat_info = {
                "glat_accu": (same_num.sum() / target_length.sum()).detach(),
                "glat_context_p": glat['context_p'],
                "glat_keep": keep_prob.mean().detach(),
                "matchmask": matchmask.detach(),
                "keep_word_mask": keep_word_mask.detach(),
                "glat_prev_output_tokens": glat_prev_output_tokens,
            }
            return glat_prev_output_tokens, glat_tgt_tokens, glat_info
        #Modify the model.forward function, enable the joint-training, which generates only one output
        #if model.args.joint_forward:
        #    j_outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat, glat_function, find_max_positions)
       # else:
        j_outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat, glat_function)
        #modify the compute_dag_loss function, to support the calculation of the joint_loss.
        losses = []
        # DAG loss
        j_losses = self._compute_joint_dag_loss(
            j_outputs["word_ins"].get("out"),
            prev_output_tokens.ne(self.task.tgt_dict.pad()),
            j_outputs["valid_start_mask"],
            j_outputs["word_ins"].get("tgt"),
            j_outputs["word_ins"].get("mask", None),
            j_outputs["n_links"],
            j_outputs["r_links"],
            name="dag-loss",
            factor=1,
            dir = 'j' ,
            matchmask= j_outputs.get('matchmask', None),
            keep_word_mask=j_outputs.get('keep_word_mask', None),
            model=model,
            glat_keep = j_outputs["glat_keep"]
        )
        '''r_losses = self._compute_dag_loss(
            r_outputs["word_ins"].get("out"),
            prev_output_tokens.ne(self.task.tgt_dict.pad()),
            r_outputs["word_ins"].get("tgt"),
            r_outputs["word_ins"].get("mask", None),
            r_outputs["links"],
            name="dag-loss",
            factor=1,
            dir = 'r' ,
            matchmask=r_outputs.get('matchmask', None),
            keep_word_mask=r_outputs.get('keep_word_mask', None),
            model=model
        )'''
        #n_length_nll_loss = n_losses.get("nll_loss", 0.0)
        #r_length_nll_loss = r_losses.get("nll_loss", 0.0)
        _losses = {"name": "loss", "loss": j_losses["loss"], "nll_loss":  j_losses["nll_loss"].detach(),
                "factor": j_losses["factor"], "ntokens": j_losses["ntokens"],  "nvalidtokens": j_losses["nvalidtokens"], "nsentences": j_losses["nsentences"],
                "loss_nofactor": j_losses["loss_nofactor"], "invalid_nsentences":j_losses["invalid_nsentences"]}
        losses += [_losses]
        j_losses["loss"] = j_losses["loss"].detach()
        dag_nll_loss = _losses["nll_loss"].detach()
        nsentences = j_losses["nsentences"]
        ntokens = j_losses["ntokens"]
        nvalidtokens = j_losses["nvalidtokens"]
        invalid_nsentences = j_losses["invalid_nsentences"]
        #length
        _losses = self._compute_loss(
            j_outputs["length"].get("out"),
            j_outputs["length"].get("tgt"),
            None,
            0,
            name="length-loss",
            factor=j_outputs["length"]["factor"], )
        losses += [_losses]
        length_nll_loss = _losses.get("nll_loss", 0.0)

        loss = sum(l["loss"] for l in losses)
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "joint_loss": j_losses["loss"],
            "dag_nll-loss": dag_nll_loss,
            "length_nll-loss": length_nll_loss.data,
            "ntokens": ntokens,
            "nvalidtokens": nvalidtokens,
            "nsentences": nsentences,
            "invalid_nsentences": invalid_nsentences,
            "sample_size": sample_size,
            "glat_acc": j_outputs.get("glat_accu", 0),
            "glat_keep": j_outputs.get("glat_keep", 0),
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss_nofactor"])
                if reduce
                else l["loss_nofactor"]
            )
        # gpu_tracker.track()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )  # each batch is 1
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))  # token-level loss

        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nvalidtokens = sum(log.get('nvalidtokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        invalid_nsentences = sum(log.get('invalid_nsentences', 0) for log in logging_outputs)
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))  # token-level loss
        glat_acc = utils.item(sum(log.get("glat_acc", 0) for log in logging_outputs))
        glat_keep = utils.item(sum(log.get("glat_keep", 0) for log in logging_outputs))

        res = {
            "ntokens": utils.item(ntokens),
            "nsentences": utils.item(nsentences),
            "nvalidtokens": utils.item(nvalidtokens),
            "invalid_nsentences": utils.item(invalid_nsentences),
            'tokens_perc': utils.item(nvalidtokens / ntokens),
            'sentences_perc': 1 - utils.item(invalid_nsentences / nsentences),
        }
        res["loss"] = loss / sample_size
        res["glat_acc"] = glat_acc / sample_size
        res["glat_keep"] = glat_keep / sample_size

        for key, value in res.items():
            metrics.log_scalar(
                key, value, sample_size, round=3
            )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = utils.item(sum(log.get(key, 0) for log in logging_outputs))
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
