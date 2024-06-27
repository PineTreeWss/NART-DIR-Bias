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

from fairseq.models.nat.fairseq_nat_model import FairseqNATModel
import logging
import random
import copy
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn, jit
import torch.nn.functional as F
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.modules import (
    PositionalEmbedding,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.nat.nonautoregressive_transformer import NATransformerDecoder
from contextlib import contextmanager
from ..custom_ops.dag_loss import torch_dag_logsoftmax_gather_inplace,torch_dag_loss
from ..custom_ops.rev_dag_loss import reverse_torch_dag_loss

logger = logging.getLogger(__name__)

@contextmanager
def torch_seed(seed):
    # modified from lunanlp
    state = torch.random.get_rng_state()
    state_cuda = torch.cuda.random.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        torch.cuda.random.set_rng_state(state_cuda)

# @jit.script
def logsumexp(x: Tensor, dim: int) -> Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))

@register_model("bi_glat_decomposed_link")
class BiGlatDecomposedLink(FairseqNATModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.init_beam_search()
        if self.args.joint_mask_fixed:
                self.valid_ral_pos_mask = eval(self.args.joint_pos_mask)
    def init_beam_search(self):
        if self.args.decode_strategy in ["beamsearch","hybrid_argmax_beam"]:
            import dag_search
            self.dag_search = dag_search
            dag_search.beam_search_init(self.args.decode_max_batchsize*self.args.argmax_token_num, self.args.decode_beamsize,
                    self.args.decode_top_cand_n, self.decoder.max_positions(), self.tgt_dict, self.args.decode_lm_path)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = BiGlatLinkDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)
        BiGlatLinkDecoder.add_args(parser)

        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            "--pred-length-offset",
            action="store_true",
            help="predicting the length difference between the target and source sentences",
        )
        parser.add_argument(
            "--sg-length-pred",
            action="store_true",
            help="stop the gradients back-propagated from the length predictor",
        )
        parser.add_argument(
            "--length-loss-factor",
            type=float,
            help="weights on the length prediction loss",
        )
        parser.add_argument(
            '--joint-forward',
            type=bool,
            default=False,
            help="Loss Forward Direction, true for Joint-Loss Training.",)
        parser.add_argument(
            '--joint-mask-fixed',
            type=bool,
            default=True,
            help="Fix the Joint training ralpos mask, true for Joint-Loss Training.",)
        parser.add_argument(
            '--joint-pos-mask',
            type=str,
            default='(0,1)',
            help="Fixed joint ralpos mask for trining.",)
        parser.add_argument(
            '--gen-link-scale',
            type=float,
            default=1.0,
            help="Inference confidence scale for bidirectional choice.",)

        parser.add_argument('--links-feature', type=str, default="feature:position", help="Features used to predict transition.")
        parser.add_argument('--max-transition-length', type=int, default=99999, help="Max transition distance. -1 means no limitation, \
                        which cannot be used for cuda custom operations. To use cuda operations with no limitation, please use a very large number such as 99999.")

        parser.add_argument("--src-upsample-scale", type=float, default=None, help="Specify the graph size with a upsample factor (lambda).  Graph Size = \\lambda * src_length")
        parser.add_argument("--src-upsample-fixed", type=int, default=None, help="Specify the graph size by a constant. Cannot use together with src-upsample-scale")
        parser.add_argument("--length-multiplier", type=float, default=None, help="Deprecated") # does not work now
        parser.add_argument('--max-decoder-batch-tokens', type=int, default=None, help="Max tokens for LightSeq Decoder when using --src-upsample-fixed")

        parser.add_argument('--filter-max-length', default=None, type=str, help='Filter the sample that above the max lengths, e.g., "128:256" indicating 128 for source, 256 for target. Default: None, for filtering according max-source-positions and max-target-positions')
        parser.add_argument("--filter-ratio", type=float, default=None, help="Deprecated") # does not work now; need support of trainer.py

        parser.add_argument('--decode-strategy', type=str, default="lookahead", help='One of "greedy", "lookahead", "viterbi", "jointviterbi", "beamsearch"')

        parser.add_argument('--decode-alpha', type=float, default=1.1, help="Used for length penalty. Beam Search finds the sentence maximize: 1 / |Y|^{alpha} [ log P(Y) + gamma log P_{n-gram}(Y)]")
        parser.add_argument('--decode-beta', type=float, default=1, help="Scale the score of logits. log P(Y, A) := sum P(y_i|a_i) + beta * sum log(a_i|a_{i-1})")
        parser.add_argument('--decode-viterbibeta', type=float, default=1, help="Length penalty for Viterbi decoding. Viterbi decoding finds the sentence maximize: P(A,Y|X) / |Y|^{beta}")
        parser.add_argument('--decode-top-cand-n', type=float, default=5, help="Numbers of top candidates when considering transition")
        parser.add_argument('--decode-gamma', type=float, default=0.1, help="Used for n-gram language model score. Beam Search finds the sentence maximize: 1 / |Y|^{alpha} [ log P(Y) + gamma log P_{n-gram}(Y)]")
        parser.add_argument('--decode-beamsize', type=float, default=100, help="Beam size")
        parser.add_argument('--argmax-token-num', type=int, default=1, help="Number of start positions for hybrid beam searching")
        #parser.add_argument('--decode-link-argmax', type=bool, default=False, help="Consider the P(link) and P(token) jointly to determine the argmax token position")
        parser.add_argument('--decode-max-beam-per-length', type=float, default=10, help="Limits the number of beam that has a same length in each step")
        parser.add_argument('--decode-top-p', type=float, default=0.9, help="Max probability of top candidates when considering transition")
        parser.add_argument('--decode-lm-path', type=str, default=None, help="Path to n-gram language model. None for not using n-gram LM")
        parser.add_argument('--decode-max-batchsize', type=int, default=32, help="Should not be smaller than the real batch size (the value is used for memory allocation)")
        parser.add_argument('--decode-dedup', type=bool, default=False, help="Use token deduplication in BeamSearch")
        
   
    def extract_valid_r_links(self, content, valid_mask):
        # batch * prelen * prelen * chunk, batch * prelen
        #print('extract')
        prelen = content.shape[1]
        translen: int = self.args.max_transition_length
        if translen > prelen - 1:
            translen = prelen - 1
        #this index aims to point the target tokens from each source token, length: prelen * prelen
        valid_links_idx = torch.arange(start=0-prelen+1, end=1, dtype=torch.long, device=content.device).unsqueeze(1) + \
                    torch.arange(translen, dtype=torch.long, device=content.device).unsqueeze(0)
        #point out the invalid index 
        #To reverse the link, we set the opposite as invalid link.
        invalid_idx_mask = ~( (valid_links_idx < valid_mask.sum(dim=-1, keepdim=True).unsqueeze(-1)) & (valid_links_idx >= 0) )
        #valid_links_idx = valid_links_idx - torch.arange(start=prelen-1, end=-1,step=-1, dtype=torch.long, device=content.device).unsqueeze(1) + 1
        valid_links_idx = valid_links_idx.unsqueeze(0).masked_fill(invalid_idx_mask, 0)
        res = content.gather(2, valid_links_idx.unsqueeze(-1).expand(-1, -1, -1, content.shape[-1]))
        res.masked_fill_(invalid_idx_mask.unsqueeze(-1), float("-inf"))
        return res, invalid_idx_mask.all(-1) # batch * prelen * trans_len * chunk, batch * prelen * trans_len

    def restore_valid_r_links(self, links):
        # batch * prelen * trans_len
        #modified
        #print('restore')
        batch_size, prelen, translen = links.shape
        translen: int = self.args.max_transition_length
        if translen > prelen - 1:
            translen = prelen - 1
        valid_links_idx = torch.arange(start=0-prelen+1, end=1, dtype=torch.long, device=links.device).unsqueeze(1) + \
                    torch.arange(translen, dtype=torch.long, device=links.device).unsqueeze(0)
        invalid_idx_mask = valid_links_idx < 0
        valid_links_idx.masked_fill_(invalid_idx_mask, prelen)
        res = torch.zeros(batch_size, prelen, prelen + 1, dtype=torch.float, device=links.device).fill_(float("-inf"))
        res.scatter_(2, valid_links_idx.unsqueeze(0).expand(batch_size, -1, -1), links)
        return res[:, :, :prelen]
    #used in decoder SAN
    def extract_valid_n_links(self, content, valid_mask):
        # batch * prelen * prelen * chunk, batch * prelen
        prelen = content.shape[1]
        translen: int = self.args.max_transition_length
        if translen > prelen - 1:
            translen = prelen - 1
        #this index aims to point the target tokens from each source token, length: prelen * prelen
        valid_links_idx = torch.arange(prelen, dtype=torch.long, device=content.device).unsqueeze(1) + \
                    torch.arange(translen, dtype=torch.long, device=content.device).unsqueeze(0) + 1
        #point out the invalid index 
        #To reverse the link, we set the opposite as invalid link.
        invalid_idx_mask = valid_links_idx >= valid_mask.sum(dim=-1, keepdim=True).unsqueeze(-1)
        valid_links_idx = valid_links_idx.unsqueeze(0).masked_fill(invalid_idx_mask, 0)
        res = content.gather(2, valid_links_idx.unsqueeze(-1).expand(-1, -1, -1, content.shape[-1]))
        res.masked_fill_(invalid_idx_mask.unsqueeze(-1), float("-inf"))
        return res, invalid_idx_mask.all(-1) # batch * prelen * trans_len * chunk, batch * prelen * trans_len
    #used in loss function
    def restore_valid_n_links(self, links):
        # batch * prelen * trans_len
        batch_size, prelen, translen = links.shape
        translen: int = self.args.max_transition_length
        if translen > prelen - 1:
            translen = prelen - 1
        valid_links_idx = torch.arange(prelen, dtype=torch.long, device=links.device).unsqueeze(1) + \
                    torch.arange(translen, dtype=torch.long, device=links.device).unsqueeze(0) + 1
        invalid_idx_mask = valid_links_idx >= prelen
        valid_links_idx.masked_fill_(invalid_idx_mask, prelen)
        res = torch.zeros(batch_size, prelen, prelen + 1, dtype=torch.float, device=links.device).fill_(float("-inf"))
        res.scatter_(2, valid_links_idx.unsqueeze(0).expand(batch_size, -1, -1), links)
        #This script enables the links value to place in the proper place!
        #Amazing!
        return res[:, :, :prelen]
    '''def extract_links(self, features, prev_output_tokens,
            link_positional,dir,query_linear, key_linear, gate_linear):

        links_feature = vars(self.args).get("links_feature", "feature:position").split(":")
        links_feature_arr = []
        if "feature" in links_feature:
            links_feature_arr.append(features)
        if "position" in links_feature or "sinposition" in links_feature:
            links_feature_arr.append(link_positional(prev_output_tokens))
        #features_withpos: batch * length * (embedding_size * 2) concat feature with positional embeddings
        features_withpos = torch.cat(links_feature_arr, dim=-1)
        batch_size = features.shape[0]
        seqlen = features.shape[1]
        chunk_num = self.args.decoder_attention_heads
        chunk_size = self.args.decoder_embed_dim // self.args.decoder_attention_heads
        ninf = float("-inf")
        target_dtype = torch.float

        query_chunks = query_linear(features_withpos).reshape(batch_size, seqlen, chunk_num, chunk_size)
        key_chunks = key_linear(features_withpos).reshape(batch_size, seqlen, chunk_num, chunk_size)
        log_gates = F.log_softmax(gate_linear(features_withpos), dim=-1, dtype=target_dtype) # batch_size * seqlen * chunk_num
        log_multi_content = (torch.einsum("bicf,bjcf->bijc", query_chunks.to(dtype=target_dtype), key_chunks.to(dtype=target_dtype)) / (chunk_size ** 0.5))
        #the code above implements a multi-head attention with linear gates.
        if self.args.max_transition_length != -1:
            if dir == 'n':
                log_multi_content_extract, link_nouse_mask = self.extract_valid_n_links(log_multi_content, prev_output_tokens.ne(self.pad))
                    # batch * seqlen * trans_len * chunk_num, batch * seqlen * trans_len
            elif dir =='r' :
                log_multi_content_extract, link_nouse_mask = self.extract_valid_r_links(log_multi_content, prev_output_tokens.ne(self.pad))
                    # batch * seqlen * trans_len * chunk_num, batch * seqlen * trans_len
            else:
                print('error: wrong direction')
                exit()
            log_multi_content_extract = log_multi_content_extract.masked_fill(link_nouse_mask.unsqueeze(-1).unsqueeze(-1), ninf)
            log_multi_content_extract = F.log_softmax(log_multi_content_extract, dim=2)
            log_multi_content_extract = log_multi_content_extract.masked_fill(link_nouse_mask.unsqueeze(-1).unsqueeze(-1), ninf)
            links = logsumexp(log_multi_content_extract + log_gates.unsqueeze(2), dim=-1) # batch_size * seqlen * trans_len
        else:
            link_mask = torch.ones(seqlen, seqlen, device=prev_output_tokens.device, dtype=bool).triu_(1).unsqueeze(0) & prev_output_tokens.ne(self.pad).unsqueeze(1)
            link_nouse_mask = link_mask.sum(dim=2, keepdim=True) == 0
            link_mask.masked_fill_(link_nouse_mask, True)
            log_multi_content.masked_fill_(~link_mask.unsqueeze(-1), ninf)
            log_multi_attention = F.log_softmax(log_multi_content, dim=2)
            log_multi_attention = log_multi_attention.masked_fill(link_nouse_mask.unsqueeze(-1), ninf)
            links = logsumexp(log_multi_attention + log_gates.unsqueeze(2), dim=-1) # batch_size * seqlen * seqlen

        return links'''
    def extract_bi_links(self, features, prev_output_tokens,
            link_positional, n_query_linear, n_key_linear, n_gate_linear, r_query_linear, r_key_linear, r_gate_linear):
        links_feature = vars(self.args).get("links_feature", "feature:position").split(":")
        links_feature_arr = []
        if "feature" in links_feature:
            links_feature_arr.append(features)
        if "position" in links_feature or "sinposition" in links_feature:
            links_feature_arr.append(link_positional(prev_output_tokens))
        #features_withpos: batch * length * (embedding_size * 2) concat feature with positional embeddings
        features_withpos = torch.cat(links_feature_arr, dim=-1)
        batch_size = features.shape[0]
        seqlen = features.shape[1]
        chunk_num = self.args.decoder_attention_heads
        chunk_size = self.args.decoder_embed_dim // self.args.decoder_attention_heads
        ninf = float("-inf")
        target_dtype = torch.float

        query_chunks = n_query_linear(features_withpos).reshape(batch_size, seqlen, chunk_num, chunk_size)
        key_chunks = n_key_linear(features_withpos).reshape(batch_size, seqlen, chunk_num, chunk_size)
        log_gates = F.log_softmax(n_gate_linear(features_withpos), dim=-1, dtype=target_dtype) # batch_size * seqlen * chunk_num
        log_multi_content = (torch.einsum("bicf,bjcf->bijc", query_chunks.to(dtype=target_dtype), key_chunks.to(dtype=target_dtype)) / (chunk_size ** 0.5))
        #the code above implements a multi-head attention with linear gates.
        if self.args.max_transition_length != -1:
            log_multi_content_extract, link_nouse_mask = self.extract_valid_n_links(log_multi_content, prev_output_tokens.ne(self.pad))
            log_multi_content_extract = log_multi_content_extract.masked_fill(link_nouse_mask.unsqueeze(-1).unsqueeze(-1), ninf)
            log_multi_content_extract = F.log_softmax(log_multi_content_extract, dim=2)
            log_multi_content_extract = log_multi_content_extract.masked_fill(link_nouse_mask.unsqueeze(-1).unsqueeze(-1), ninf)
            n_links = logsumexp(log_multi_content_extract + log_gates.unsqueeze(2), dim=-1) # batch_size * seqlen * trans_len
        else:
            link_mask = torch.ones(seqlen, seqlen, device=prev_output_tokens.device, dtype=bool).triu_(1).unsqueeze(0) & prev_output_tokens.ne(self.pad).unsqueeze(1)
            link_nouse_mask = link_mask.sum(dim=2, keepdim=True) == 0
            link_mask.masked_fill_(link_nouse_mask, True)
            log_multi_content.masked_fill_(~link_mask.unsqueeze(-1), ninf)
            log_multi_attention = F.log_softmax(log_multi_content, dim=2)
            log_multi_attention = log_multi_attention.masked_fill(link_nouse_mask.unsqueeze(-1), ninf)
            n_links = logsumexp(log_multi_attention + log_gates.unsqueeze(2), dim=-1) # batch_size * seqlen * seqlen

        #extract r links
        query_chunks = r_query_linear(features_withpos).reshape(batch_size, seqlen, chunk_num, chunk_size)
        key_chunks = r_key_linear(features_withpos).reshape(batch_size, seqlen, chunk_num, chunk_size)
        log_gates = F.log_softmax(r_gate_linear(features_withpos), dim=-1, dtype=target_dtype) # batch_size * seqlen * chunk_num
        log_multi_content = (torch.einsum("bicf,bjcf->bijc", query_chunks.to(dtype=target_dtype), key_chunks.to(dtype=target_dtype)) / (chunk_size ** 0.5))
        #the code above implements a multi-head attention with linear gates.
        if self.args.max_transition_length != -1:
            log_multi_content_extract, link_nouse_mask = self.extract_valid_r_links(log_multi_content, prev_output_tokens.ne(self.pad))
            log_multi_content_extract = log_multi_content_extract.masked_fill(link_nouse_mask.unsqueeze(-1).unsqueeze(-1), ninf)
            log_multi_content_extract = F.log_softmax(log_multi_content_extract, dim=2)
            log_multi_content_extract = log_multi_content_extract.masked_fill(link_nouse_mask.unsqueeze(-1).unsqueeze(-1), ninf)
            r_links = logsumexp(log_multi_content_extract + log_gates.unsqueeze(2), dim=-1) # batch_size * seqlen * trans_len
        else:
            link_mask = torch.ones(seqlen, seqlen, device=prev_output_tokens.device, dtype=bool).triu_(1).unsqueeze(0) & prev_output_tokens.ne(self.pad).unsqueeze(1)
            link_nouse_mask = link_mask.sum(dim=2, keepdim=True) == 0
            link_mask.masked_fill_(link_nouse_mask, True)
            log_multi_content.masked_fill_(~link_mask.unsqueeze(-1), ninf)
            log_multi_attention = F.log_softmax(log_multi_content, dim=2)
            log_multi_attention = log_multi_attention.masked_fill(link_nouse_mask.unsqueeze(-1), ninf)
            r_links = logsumexp(log_multi_attention + log_gates.unsqueeze(2), dim=-1) # batch_size * seqlen * seqlen
        return n_links,r_links
    def extract_features(self, prev_output_tokens, encoder_out, rand_seed, link_dir,require_links=False):
        if link_dir == 'b':
            with torch_seed(rand_seed):
                features, _ = self.decoder.extract_features(
                    prev_output_tokens,
                    encoder_out=encoder_out,
                    embedding_copy=False
                )
                # word_ins_out = self.decoder.output_layer(features)
                word_ins_out = self.decoder.output_projection(features)
                links = None
                if require_links:
                    n_links,r_links = self.extract_bi_links(features, \
                                prev_output_tokens, \
                                self.decoder.link_positional, \
                                self.decoder.n_link_query_linear, \
                                self.decoder.n_link_key_linear, \
                                self.decoder.n_link_gate_linear, \
                                self.decoder.r_link_query_linear, \
                                self.decoder.r_link_key_linear, \
                                self.decoder.r_link_gate_linear, \
                            )
            return word_ins_out, n_links,r_links
        '''elif link_dir == 'n':
            with torch_seed(rand_seed):
                features, _ = self.decoder.extract_features(
                    prev_output_tokens,
                    encoder_out=encoder_out,
                    embedding_copy=False
                )
                # word_ins_out = self.decoder.output_layer(features)
                word_ins_out = self.decoder.output_projection(features)
                links = None
                if require_links:
                    n_links = self.extract_links(features, \
                                prev_output_tokens, \
                                self.decoder.link_positional, \
                                link_dir, \
                                self.decoder.n_link_query_linear, \
                                self.decoder.n_link_key_linear, \
                                self.decoder.n_link_gate_linear, \
                            )
            return word_ins_out, n_links
        elif link_dir == 'r':
            with torch_seed(rand_seed):
                features, _ = self.decoder.extract_features(
                    prev_output_tokens,
                    encoder_out=encoder_out,
                    embedding_copy=False
                )
                # word_ins_out = self.decoder.output_layer(features)
                word_ins_out = self.decoder.output_projection(features)
                links = None
                if require_links:
                    r_links = self.extract_links(features, \
                                prev_output_tokens, \
                                self.decoder.link_positional, \
                                link_dir, \
                                self.decoder.r_link_query_linear, \
                                self.decoder.r_link_key_linear, \
                                self.decoder.r_link_gate_linear, \
                            )
            return word_ins_out, r_links'''
    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat=None, glat_function=None, find_max_positions=None, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )
        rand_seed = random.randint(0, 19260817)
        if self.args.joint_forward:
            if self.args.joint_mask_fixed:
                valid_ral_pos_mask = self.valid_ral_pos_mask
            else:
                valid_ral_pos_mask = [0,1]
            bsz = tgt_tokens.shape[0]
            token_positions = ~(tgt_tokens.eq(self.pad) | tgt_tokens.eq(self.bos) | tgt_tokens.eq(self.eos))
            target_length = (token_positions).sum(1)
            output_length = prev_output_tokens.ne(self.pad).sum(1)
            valid_tgt_start = (target_length * valid_ral_pos_mask[0]+1).clone().detach().type(torch.int16)
            valid_tgt_end = (target_length * valid_ral_pos_mask[1]+1).clone().detach().type(torch.int16)
            valid_tgt_pos_mask = torch.zeros(size=tgt_tokens.shape,out=None,  dtype=torch.bool,layout=torch.strided,device=tgt_tokens.device,requires_grad=False).detach()
            for i in range(bsz):
                valid_tgt_pos_mask[i][valid_tgt_start[i]:valid_tgt_end[i]] = True
            if glat and tgt_tokens is not None:
                with torch.set_grad_enabled(glat.get('require_glance_grad', False)):
                    word_ins_out, n_links, r_links = self.extract_features(prev_output_tokens, encoder_out,  rand_seed, link_dir='b', require_links=True)
                    prev_output_tokens, tgt_tokens, glat_info = glat_function(self, word_ins_out, tgt_tokens, prev_output_tokens, glat,valid_tgt_pos_mask, dir='j',links=(n_links,r_links))
            word_ins_out, n_links, r_links = self.extract_features(prev_output_tokens, encoder_out, rand_seed, link_dir = 'b', require_links=True)
            #print('output r_links')
            #print(r_links[51])
            #exit()
            ret = {
                "word_ins": {
                    "out": word_ins_out,
                    "tgt": tgt_tokens,
                    "mask": tgt_tokens.ne(self.pad),
                    "nll_loss": True,
                }
            }
            ret['n_links'] = n_links
            ret['r_links'] = r_links

            ret["length"] = {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            }
            ret['valid_start_mask'] = valid_tgt_pos_mask
            if glat_info is not None:
                ret.update(glat_info)
            return ret
        else:
        # decoding
            glat_info_n = None
            glat_info_r = None
            if glat and tgt_tokens is not None:
                with torch.set_grad_enabled(glat.get('require_glance_grad', False)):
                    word_ins_out, n_links, r_links = self.extract_features(prev_output_tokens, encoder_out, rand_seed, link_dir='b',require_links=True)
                    prev_output_tokens_n, tgt_tokens_n, glat_info_n = glat_function(self, word_ins_out, tgt_tokens, prev_output_tokens, glat, dir='n',links=n_links)
                    prev_output_tokens_r, tgt_tokens_r, glat_info_r = glat_function(self, word_ins_out, tgt_tokens, prev_output_tokens, glat, dir='r',links=r_links)
                    word_ins_out = None
            word_ins_out, n_links, r_links = self.extract_features(prev_output_tokens_n, encoder_out, rand_seed, link_dir = 'b', require_links=True)
            r_links = r_links.detach()
            n_ret = {
                "word_ins": {
                    "out": word_ins_out,
                    "tgt": tgt_tokens_n,
                    "mask": tgt_tokens_n.ne(self.pad),
                    "nll_loss": True,
                }
            }
            n_ret['links'] = n_links

            n_ret["length"] = {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            }
            if glat_info_n is not None:
                n_ret.update(glat_info_n)
            word_ins_out, n_links2, r_links2 = self.extract_features(prev_output_tokens_r, encoder_out, rand_seed, link_dir = 'b', require_links=True)
            n_links2 = n_links2.detach()
            r_ret = {
                "word_ins": {
                    "out": word_ins_out,
                    "tgt": tgt_tokens_r,
                    "mask": tgt_tokens_r.ne(self.pad),
                    "nll_loss": True,
                }
            }
            r_ret['links'] = r_links2

            r_ret["length"] = {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            }
            if glat_info_r is not None:
                r_ret.update(glat_info_r)
            return n_ret,r_ret


    def initialize_output_tokens_with_length(self, src_tokens, length_tgt):
        max_length = length_tgt.max()
        idx_length = utils.new_arange(src_tokens, max_length)
        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)
        return initial_output_tokens

    def initialize_output_tokens_upsample_by_tokens(self, src_tokens):
        if vars(self.args).get("src_upsample_scale", None) is not None:
            length_tgt = torch.sum(src_tokens.ne(self.tgt_dict.pad_index), -1)
            length_tgt = (length_tgt * self.args.src_upsample_scale).long().clamp_(min=2)
        else:
            length_tgt = torch.zeros(src_tokens.shape[0], device=src_tokens.device, dtype=src_tokens.dtype).fill_(self.args.src_upsample_fixed)
        return self.initialize_output_tokens_with_length(src_tokens, length_tgt)

    def initialize_output_tokens_multiplier_by_tokens(self, src_tokens, tgt_tokens):
        length_tgt = torch.sum(tgt_tokens.ne(self.tgt_dict.pad_index), -1)
        length_tgt = (length_tgt * self.args.length_multiplier).long().clamp_(min=2)
        return self.initialize_output_tokens_with_length(src_tokens, length_tgt)

    def initialize_output_tokens_by_tokens(self, src_tokens, tgt_tokens):
        if vars(self.args).get("src_upsample_scale", None) is not None or vars(self.args).get("src_upsample_fixed", None) is not None:
            return self.initialize_output_tokens_upsample_by_tokens(src_tokens)
        elif vars(self.args).get("length_multiplier", None) is not None:
            return self.initialize_output_tokens_multiplier_by_tokens(src_tokens, tgt_tokens)

    def initialize_output_tokens_upsample(self, encoder_out, src_tokens):
        # length prediction
        if vars(self.args).get("src_upsample_scale", None) is not None:
            length_tgt = torch.sum(src_tokens.ne(self.tgt_dict.pad_index), -1)
            length_tgt = (length_tgt * self.args.src_upsample_scale).long().clamp_(min=2)
        else:
            length_tgt = torch.zeros(src_tokens.shape[0], device=src_tokens.device, dtype=src_tokens.dtype).fill_(self.args.src_upsample_fixed)
        initial_output_tokens = self.initialize_output_tokens_with_length(src_tokens, length_tgt)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def initialize_output_tokens_multiplier(self, encoder_out, src_tokens):
        # length prediction
        length_tgt = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
        )
        length_tgt = (length_tgt * self.args.length_multiplier).long().clamp_(min=2)
        initial_output_tokens = self.initialize_output_tokens_with_length(src_tokens, length_tgt)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens):
        if vars(self.args).get("src_upsample_scale", None) is not None or vars(self.args).get("src_upsample_fixed", None) is not None:
            return self.initialize_output_tokens_upsample(encoder_out, src_tokens)
        elif vars(self.args).get("length_multiplier", None) is not None:
            return self.initialize_output_tokens_multiplier(encoder_out, src_tokens)

    def max_positions(self):
        if vars(self.args).get("filter_max_length", None) is not None:
            if ":" not in self.args.filter_max_length:
                a = b = int(self.args.filter_max_length)
            else:
                a, b = self.args.filter_max_length.split(":")
                a, b = int(a), int(b)
            return (a, b)
        else:
            if vars(self.args).get("src_upsample_fixed", None) is not None:
                return (self.encoder.max_positions(), self.decoder.max_positions())
            elif vars(self.args).get("src_upsample_scale", None) is not None:
                return (min(self.encoder.max_positions(), int(self.decoder.max_positions() / self.args.src_upsample_scale)), self.decoder.max_positions())
            else:
                return (min(self.encoder.max_positions(), int(self.decoder.max_positions() / self.args.length_multiplier)), self.decoder.max_positions())
    def _reverse_search(self,dagscore,nextstep_idx,token_logits_idx,tgt_length):
        #Revert the beam search information for the reverse beam search
        batchsize,prelen,topk = dagscore.shape
        reverse_dagscore = dagscore.clone()
        reverse_nextstep_idx = nextstep_idx.clone()
        reverse_token_logits_idx = token_logits_idx.clone()
        #掉个的时候就没考虑tarlen，其实只要把所有的idx按照tarlen调过来就好了啊。脱裤子放屁�?        
        for b in range(batchsize):
            for step in range(tgt_length[b]):
                reverse_dagscore[b][step] = dagscore[b][tgt_length[b]-step-1]
                reverse_token_logits_idx[b][step] = token_logits_idx[b][tgt_length[b]-step-1]
                #eos_mask = reverse_token_logits_idx[b][step] == self.tgt_dict.eos()
                #bos_mask = reverse_token_logits_idx[b][step] == self.tgt_dict.bos()
                '''if True in (eos_mask):
                    reverse_token_logits_idx[b][step].masked_fill(mask=eos_mask,value = self.tgt_dict.bos())
                if True in (bos_mask):
                    reverse_token_logits_idx[b][step].masked_fill(mask=bos_mask,value = self.tgt_dict.eos())'''
                reverse_nextstep_idx[b][step] = tgt_length[b]-nextstep_idx[b][tgt_length[b]-step-1]-1
        return reverse_dagscore,reverse_nextstep_idx,reverse_token_logits_idx
    def _reverse_search_output(self,output_tokens,output_scores):
        #Reverse the Beam output for DAG Search
        return torch.flip(output_tokens,dims=[1]),torch.flip(output_scores,dims=[1])
    def __bidir_dp(self,targets,pack,direction):
        outputs = pack['output_logits']
        prelen = outputs.shape[1]
        outputs, match_all = torch_dag_logsoftmax_gather_inplace(outputs, targets.unsqueeze(1).expand(-1, prelen, -1))
        match_all = match_all.transpose(1, 2)            
        #print('match_all: ')
        #print(match_all)
        output_length = pack['output_masks'].sum(dim=-1)
        #print('output_length:')
        #print(output_length)
        if direction == 'n':
            target_length = pack['n_tgt_mask'].sum(dim=-1)
            #print('target_length: ')
            #print(target_length)
            loss_result = torch_dag_loss(match_all, pack['n_links'], output_length, target_length)
            loss_result /= match_all.shape[2]
        elif direction == 'r':
            target_length = pack['r_tgt_mask'].sum(dim=-1)
            #print('target_length: ')
            #print(target_length)
            #print(pack['r_links'])
            #exit()
            loss_result = reverse_torch_dag_loss(match_all,pack['r_links'], output_length, target_length)
            loss_result /= output_length
        return loss_result.detach()
    def _bidir_argmax_selection(self,n_logits,r_logits,output_tokens_forward,output_tokens_backward,mode='normal',bi_select_pack=0):
        bsz,_ = output_tokens_forward.shape
        output_tokens = []
        output_scores = []
        #print('decode_logits')
        #print(n_logits)
        #print(r_logits)
        if mode == 'bi-select':
            n_dp_logits = self.__bidir_dp(output_tokens_forward,bi_select_pack,'n')
            r_dp_logits = self.__bidir_dp(output_tokens_forward,bi_select_pack,'r')
            n_logits = (n_dp_logits + r_dp_logits)
            n_dp_logits = self.__bidir_dp(output_tokens_backward,bi_select_pack,'n')
            r_dp_logits = self.__bidir_dp(output_tokens_backward,bi_select_pack,'r')
            r_logits = (n_dp_logits + r_dp_logits)
            #print('dp logits')
            #print(n_logits)
            #print(r_logits)
        for batch in range(bsz):
            if mode=='beam':
                tmp_n_logits = n_logits[batch][0] * self.args.gen_link_scale
                tmp_r_logits = r_logits[batch][0]
            else:
                tmp_n_logits = -n_logits[batch] * self.args.gen_link_scale
                tmp_r_logits = -r_logits[batch]
            if tmp_n_logits > tmp_r_logits:
                #print('n',tmp_n_logits,tmp_r_logits)
                output_tokens.append(output_tokens_forward[batch].tolist())
                output_scores.append(n_logits[batch])
            else:
                #print('r',tmp_n_logits,tmp_r_logits)
                output_tokens.append(output_tokens_backward[batch].tolist())
                output_scores.append(r_logits[batch])
        #output_tokens = torch.tensor(output_tokens, device=output_tokens_forward.device,dtype = output_tokens_forward.dtype)
        #output_scores = torch.tensor(output_scores, device=output_tokens_forward.device,dtype=torch.float)
        #print(output_tokens.shape)
        #print(output_scores.shape)
        #print(output_scores)
        #exit()
        return output_tokens,output_scores
    def _pad_batch_samples(self,torch_device,output_length,batch_dagscores_sample_l,batch_dagscores_sample_r,batch_nextstep_idx_l,
            batch_nextstep_idx_r,batch_logit_idx_l,batch_logit_idx_r,batch_output_length_l,batch_output_length_r):
        #Pad the samples in a specific batch
        def __pad_tool(samples,pad_index,torch_dtype,pad_mode = 'none'):
                if pad_mode == 'sequence_len':
                    output_seqlen = max([len(res) for res in samples])
                    output_length = pad_index
                    output_samples = []
                    #print(output_length.shape)
                    for i in range(len(samples)):
                        output_samples.append([samples[i].tolist() + [[int(output_length[int(i/self.args.argmax_token_num)]-1)] * len(samples[0][0])] *(output_seqlen - len(samples[i]))])
                    output_samples = torch.tensor(output_samples, device=torch_device)
                    return output_samples
                else:
                    output_seqlen = max([len(res) for res in samples])
                    output_samples = [res.tolist() + [pad_index] * (output_seqlen - len(res)) for res in samples]
                    output_samples = torch.tensor(output_samples, device=torch_device,dtype=torch_dtype)
                return output_samples
        def __reverse_tool(samples,reverse_length,reverse_idx=False):
            reverse_samples = []
            for i in range(len(samples)):
                sample = samples[i]
                reverse_sample = []
                sample_reverse_length = reverse_length[i]
                for j in range(len(sample)):
                    if j < sample_reverse_length:
                        if(reverse_idx):
                            reverse_sample.append((sample_reverse_length-1-sample[sample_reverse_length-j-1]).tolist())
                        else:
                            reverse_sample.append(sample[sample_reverse_length-j-1].tolist())
                    else:
                        reverse_sample.append(sample[j].tolist())
                reverse_samples.append(reverse_sample)
            return torch.tensor(reverse_samples, device=torch_device)
        def __move_tool(samples,sample_length,target_length):
            for i in range(len(samples)):
                sample = samples[i]
                for j in range(len(sample)):
                    if j < sample_length[i]-1:
                        sample[j] -= (target_length[int(i/self.args.argmax_token_num)] - sample_length[i])
            return samples
        batch_dagscores_sample_l = __pad_tool(batch_dagscores_sample_l,[float('-inf')]* self.args.decode_top_cand_n,torch_dtype=torch.float)
        batch_dagscores_sample_r = __pad_tool(batch_dagscores_sample_r,[float('-inf')]* self.args.decode_top_cand_n,torch_dtype=torch.float)
        batch_nextstep_idx_l = __pad_tool(batch_nextstep_idx_l,[0] * self.args.decode_top_cand_n,torch_dtype=torch.int32)
        batch_nextstep_idx_r = __pad_tool(batch_nextstep_idx_r,[0] * self.args.decode_top_cand_n,torch_dtype=torch.int32)
        batch_nextstep_idx_r = __move_tool(batch_nextstep_idx_r,batch_output_length_r,output_length)
        batch_logit_idx_l = __pad_tool(batch_logit_idx_l,[self.tgt_dict.pad_index] * self.args.decode_top_cand_n,torch_dtype=torch.int32)
        batch_logit_idx_r = __pad_tool(batch_logit_idx_r,[self.tgt_dict.pad_index] * self.args.decode_top_cand_n,torch_dtype=torch.int32)
        batch_dagscores_sample_l = __reverse_tool(batch_dagscores_sample_l,batch_output_length_l)
        batch_nextstep_idx_l = __reverse_tool(batch_nextstep_idx_l,batch_output_length_l,reverse_idx=True)
        batch_logit_idx_l = __reverse_tool(batch_logit_idx_l,batch_output_length_l)
        return batch_dagscores_sample_l,batch_dagscores_sample_r,batch_nextstep_idx_l,batch_nextstep_idx_r,batch_logit_idx_l,batch_logit_idx_r,batch_output_length_l,batch_output_length_r
    def _beam_search_pack(self,batch_dagscores_sample_l,batch_nextstep_idx_l,batch_logit_idx_l,batch_output_length_l,batch_dagscores_sample_r,batch_nextstep_idx_r,batch_logit_idx_r,batch_output_length_r,start_token_position,start_token_idx,start_token_logits,pad_length):
        def __filer_samples(dagscores,nextstep_idx,logit_idx,output_length):
            filter_dagscores = []
            filter_nextstep_idx = []
            filter_logit_idx = []
            filter_output_length = []
            filter_dict = {}
            next_id = 0
            for i in range(len(dagscores)):
                if torch.all(nextstep_idx[i]==0):
                    filter_dict[i] = 'Removed'
                else:
                    filter_dict[i] = next_id
                    next_id += 1
                    filter_dagscores.append(dagscores[i].cpu().detach().numpy())
                    filter_nextstep_idx.append(nextstep_idx[i].cpu().detach().numpy().astype(np.int32))
                    filter_logit_idx.append(logit_idx[i].cpu().detach().numpy().astype(np.int32))
                    filter_output_length.append(output_length[i].astype(np.int32))
            return filter_dagscores,filter_nextstep_idx,filter_logit_idx,filter_output_length,filter_dict
        def __restore_samples(l_res,l_score,l_dict,r_res,r_score,r_dict,origin_length,start_token_position,start_token_idx,start_token_logits,batch_output_length_l,batch_output_length_r):
            #Calculate the logits for each sample candidate, reform the sample with highest logits
            sample_num,token_num = start_token_logits.shape
            final_res = []
            final_score = []
            for sample_id in range(sample_num):
                sample_candidate_logit = []
                for token_id in range(token_num):
                    sentence_logit = 0
                    best_token_logit = start_token_logits[sample_id][token_id]
                    batch_sample_id = sample_id * token_num + token_id
                    left_logit = l_dict[batch_sample_id]
                    if(left_logit) == 'Removed':
                        #print('---------------------------------------LEFT REMOVED!---------------------------------------')
                        left_logit = 0
                        left_length = 0
                        #print(batch_output_length_r[batch_sample_id])
                    else:
                        #print('l_res: ')
                       # print(l_res[left_logit])
                        left_length = np.count_nonzero(l_res[left_logit]!=self.tgt_dict.pad())
                        left_length -= np.count_nonzero(l_res[left_logit]==self.tgt_dict.bos())
                        #left_length -= np.count_nonzero(l_res[left_logit]==self.tgt_dict.eos())
                        #print('left length:',left_length)
                        #print(l_res[left_logit])
                        #print(type(l_res[left_logit]))
                        left_logit = l_score[left_logit] * left_length
                        
                    right_logit = r_dict[batch_sample_id]
                    if(right_logit) == 'Removed':
                        #print('-----------------------------------------RIGHT REMOVED!-------------------------------------')
                        right_logit = 0
                        right_length = 0
                        #print(batch_output_length_l[batch_sample_id])
                    else:
                        #print('r_res: ')
                        #print(r_res[right_logit])
                        right_length = np.count_nonzero(r_res[right_logit]!=self.tgt_dict.pad())
                        right_length -= np.count_nonzero(r_res[right_logit]==self.tgt_dict.bos())
                        #right_length -= np.count_nonzero(r_res[right_logit]==self.tgt_dict.eos())
                        #print('right length:',right_length)
                        #print(r_res[right_logit])
                        #print(type(r_res[right_logit]))
                        right_logit = r_score[right_logit] * right_length
                    sample_candidate_logit.append((float(best_token_logit.cpu().numpy()) + left_logit + right_logit)/(left_length+right_length+1))
                    #print('best token logit:',float(best_token_logit.cpu().numpy()))
                    #print('left logit:',left_logit)
                    #print('right logit:',right_logit)
                    
                max_value_idx = np.argmax(np.array(sample_candidate_logit))
                #print(sample_candidate_logit)
                #print(max_value_idx)
                #Restore the sample
                batch_sample_id = sample_id * token_num + max_value_idx
                left_idx = l_dict[batch_sample_id]
                right_idx = r_dict[batch_sample_id]
                sample_score = sample_candidate_logit[max_value_idx]
                best_token_id = start_token_idx[sample_id][max_value_idx]
                sample_res = []
                if(left_idx) != 'Removed':
                    sample_res = np.flip(l_res[left_idx][0:(batch_output_length_l[left_idx]-1)]).tolist()
                sample_res.append(int(best_token_id.cpu().numpy()))
                if (right_idx) != 'Removed':
                    sample_res = sample_res + r_res[right_idx][0:(batch_output_length_r[right_idx]-1)].tolist()
                if len(sample_res) != pad_length:
                    sample_res = sample_res + [self.tgt_dict.pad_index] * (pad_length - len(sample_res))
                final_res.append(sample_res)
                final_score.append(sample_score)
            return final_res,final_score
        origin_length = len(batch_dagscores_sample_l)
        batch_dagscores_sample_r,batch_nextstep_idx_r,batch_logit_idx_r,batch_output_length_r,filter_dict_r = __filer_samples(batch_dagscores_sample_r,batch_nextstep_idx_r,
                                                                                                                batch_logit_idx_r,batch_output_length_r)
        batch_dagscores_sample_l,batch_nextstep_idx_l,batch_logit_idx_l,batch_output_length_l,filter_dict_l = __filer_samples(batch_dagscores_sample_l,batch_nextstep_idx_l,
                                                                                                                batch_logit_idx_l,batch_output_length_l)
        l_dagscores = np.ascontiguousarray(batch_dagscores_sample_l)
        l_nextstep_idx = np.ascontiguousarray(batch_nextstep_idx_l)
        l_logits_idx = np.ascontiguousarray(batch_logit_idx_l)
        l_output_length_cpu = np.ascontiguousarray(batch_output_length_l)
        r_dagscores = np.ascontiguousarray(batch_dagscores_sample_r)
        r_nextstep_idx = np.ascontiguousarray(batch_nextstep_idx_r)
        r_logits_idx = np.ascontiguousarray(batch_logit_idx_r)
        r_output_length_cpu = np.ascontiguousarray(batch_output_length_r)
        '''print(r_dagscores.dtype)
        print(r_nextstep_idx.dtype)
        print(r_logits_idx.dtype)
        print(r_output_length_cpu.dtype)
        print(l_dagscores.dtype)
        print(l_nextstep_idx.dtype)
        print(l_logits_idx.dtype)
        print(l_output_length_cpu.dtype)
        for i in range(len(r_dagscores)):
            print(r_dagscores[i])
            print(r_nextstep_idx[i])
            print(r_logits_idx[i])
            print(r_output_length_cpu[i])'''
        r_res, r_score = self.dag_search.dag_search(r_dagscores, r_nextstep_idx, r_logits_idx,
                    r_output_length_cpu,
                    self.args.decode_alpha,
                    self.args.decode_gamma,
                    self.args.decode_beamsize,
                    self.args.decode_max_beam_per_length,
                    self.args.decode_top_p,
                    self.tgt_dict.pad_index,
                    self.tgt_dict.bos_index,
                    1 if self.args.decode_dedup else 0
                )
        l_res, l_score = self.dag_search.dag_search(l_dagscores, l_nextstep_idx, l_logits_idx,
                    l_output_length_cpu,
                    self.args.decode_alpha,
                    self.args.decode_gamma,
                    self.args.decode_beamsize,
                    self.args.decode_max_beam_per_length,
                    self.args.decode_top_p,
                    self.tgt_dict.pad_index,
                    self.tgt_dict.bos_index,
                    1 if self.args.decode_dedup else 0
                )
        res,score = __restore_samples(l_res,l_score,filter_dict_l,r_res,r_score,filter_dict_r,origin_length,start_token_position,start_token_idx,start_token_logits,batch_output_length_l,batch_output_length_r)
        return res,score
    def _argmax_beam_samples(self,  n_dagscores,r_dagscores, n_nextstep_idx, r_nextstep_idx, n_logits_idx, r_logits_idx, output_length, start_token_position,start_token_idx,start_token_logits):
        #Aggregate the samples for the bidirectional beam search.
        #start token position: bsz * start_pos_num
        #dagscores,nextstep_idx, logits_idx: bsz * prelen * topk
        #output_length: [bsz]
        #returns two groups of samples, used for bidirectional beam-search.
        bsz,start_pos_num = start_token_position.shape
        _,prelen,topk = n_dagscores.shape
        beam_pad_length = prelen
        batch_dagscores_sample_r = []
        batch_dagscores_sample_l = []
        batch_nextstep_idx_r = []
        batch_nextstep_idx_l = []
        batch_logit_idx_r = []
        batch_logit_idx_l = []
        batch_output_length_r = []
        batch_output_length_l = []
        batch_best_token_pos = []
        batch_best_token_idx = []
        for batch in range(bsz):
            for i in range(start_pos_num):
                token = start_token_position[batch][i]
                token_id = start_token_idx[batch][i]
                #Generate right sample
                if token < output_length[batch]-1:
                    r_dagscores_sample = n_dagscores[batch][token:output_length[batch]].cpu().detach().numpy()
                    r_nextstep_idx_sample = n_nextstep_idx[batch][token:output_length[batch]].cpu().detach().numpy()
                    r_logits_idx_sample = n_logits_idx[batch][token:output_length[batch]].cpu().detach().numpy()
                    r_output_length_sample = np.int32(output_length[batch].cpu() - token.cpu())
                else:
                    r_dagscores_sample = np.array([])
                    r_nextstep_idx_sample = np.array([])
                    r_logits_idx_sample = np.array([])
                    r_output_length_sample = np.int32(0)
                    #initiate null samples for padding.
                if token > 0:
                    l_dagscores_sample = r_dagscores[batch][:token+1].cpu().detach().numpy()
                    l_nextstep_idx_sample = r_nextstep_idx[batch][:token+1].cpu().detach().numpy()
                    l_logits_idx_sample = r_logits_idx[batch][:token+1].cpu().detach().numpy()
                    l_output_length_sample = np.int32(token.cpu() + 1)
                else:
                    l_dagscores_sample = np.array([])
                    l_nextstep_idx_sample = np.array([])
                    l_logits_idx_sample = np.array([])
                    l_output_length_sample = np.int32(0)
                #each token generate two samples: left search & right search. append those samples to the batch list.
                '''print('original_dag_scores:')
                print(n_dagscores[batch])
                print(r_dagscores[batch])
                print(n_dagscores[batch].shape)
                print('original_logit_idx:')
                print(n_logits_idx[batch])
                print('r:')
                print(r_logits_idx[batch])
                print('original_output_length:')
                print(output_length[batch])
                print('original_output_idx')
                print(n_nextstep_idx[batch])
                print('r:')
                print(r_nextstep_idx[batch])
                print('l_dag_scores:')
                print(l_dagscores_sample.shape)
                print(l_dagscores_sample)
                print('r_dag_scores:')
                print(r_dagscores_sample)
                print('token_pos:')
                print(token)
                print('best_token_idx:')
                print(token_id)
                print('logit_idx_l:')
                print(l_logits_idx_sample)
                print('logit_idx_r:')
                print(r_logits_idx_sample)
                print('nextstep_idx_l:')
                print(l_nextstep_idx_sample)
                print('nextstep_idx_r:')
                print(r_nextstep_idx_sample)
                print('output_length_l:')
                print(l_output_length_sample)
                print('output_length_r:')
                print(r_output_length_sample)
                exit()'''
                batch_dagscores_sample_l.append(l_dagscores_sample)
                batch_dagscores_sample_r.append(r_dagscores_sample)
                batch_best_token_pos.append(token)
                batch_best_token_idx.append(token_id)
                batch_logit_idx_l.append(l_logits_idx_sample)
                batch_logit_idx_r.append(r_logits_idx_sample)
                batch_nextstep_idx_l.append(l_nextstep_idx_sample)
                batch_nextstep_idx_r.append(r_nextstep_idx_sample)
                batch_output_length_l.append(l_output_length_sample)
                batch_output_length_r.append(r_output_length_sample)
        batch_dagscores_sample_l,batch_dagscores_sample_r,batch_nextstep_idx_l,
        batch_dagscores_sample_l,batch_dagscores_sample_r,batch_nextstep_idx_l,batch_nextstep_idx_r,batch_logit_idx_l,batch_logit_idx_r,batch_output_length_l,batch_output_length_r = self._pad_batch_samples(n_logits_idx[0].device,output_length,batch_dagscores_sample_l,batch_dagscores_sample_r,batch_nextstep_idx_l,
        batch_nextstep_idx_r,batch_logit_idx_l,batch_logit_idx_r,batch_output_length_l,batch_output_length_r)
        res,score = self._beam_search_pack(batch_dagscores_sample_l,batch_nextstep_idx_l,batch_logit_idx_l,batch_output_length_l,batch_dagscores_sample_r,batch_nextstep_idx_r,batch_logit_idx_r,batch_output_length_r,start_token_position,start_token_idx,start_token_logits,beam_pad_length)
        return res,score
        #return dagscores,nextstep_idx,logits_idx,output_length
    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens

        history = decoder_out.history
        rand_seed = random.randint(0, 19260817)

        # execute the decoder
        output_logits, n_links,r_links = self.extract_features(output_tokens, encoder_out, rand_seed,link_dir='b',require_links=True)
        if self.args.max_transition_length != -1:
            n_links = self.restore_valid_n_links(n_links)
            r_links = self.restore_valid_r_links(r_links)
        output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1)

        output_logits_normalized = output_logits.log_softmax(dim=-1)
        unreduced_logits, unreduced_tokens = output_logits_normalized.max(dim=-1)
        invalid_tokens_mask = ~(unreduced_tokens.ne(self.tgt_dict.pad_index) * unreduced_tokens.ne(self.tgt_dict.bos()) * unreduced_tokens.ne(self.tgt_dict.eos()))
        bsz = unreduced_tokens.shape[0]
        token_positions = ~(unreduced_tokens.eq(self.pad))
        target_length = (token_positions).sum(1)
        valid_ral_pos_mask = self.valid_ral_pos_mask
        valid_tgt_start = (target_length * valid_ral_pos_mask[0]+0.9).clone().detach().type(torch.int16)
        valid_tgt_end = (target_length * valid_ral_pos_mask[1]+0.9).clone().detach().type(torch.int16)
        valid_tgt_pos_mask = torch.zeros(size=unreduced_tokens.shape,out=None,  dtype=torch.bool,layout=torch.strided,device=unreduced_tokens.device,requires_grad=False)
        for i in range(bsz):
            valid_tgt_pos_mask[i][valid_tgt_start[i]:valid_tgt_end[i]] = True
        unreduced_tokens = unreduced_tokens.tolist()
        if self.args.decode_strategy in ["lookahead", "greedy"]:
            if self.args.decode_direction == "forward":
                links = n_links
                if self.args.decode_strategy == "lookahead":
                    output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                    links_idx = (links + unreduced_logits.unsqueeze(1) * self.args.decode_beta).max(dim=-1)[1].cpu().tolist() # batch * prelen
                elif self.args.decode_strategy == "greedy":
                    output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                    links_idx = links.max(dim=-1)[1].cpu().tolist() # batch * prelen

                unpad_output_tokens = []
                for i, length in enumerate(output_length):
                    last = unreduced_tokens[i][0]
                    j = 0
                    res = [last]
                    while j != length - 1:
                        j = links_idx[i][j]
                        now_token = unreduced_tokens[i][j]
                        if now_token != self.tgt_dict.pad_index and now_token != last:
                            res.append(now_token)
                        last = now_token
                    unpad_output_tokens.append(res)

                output_seqlen = max([len(res) for res in unpad_output_tokens])
                output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in unpad_output_tokens]
                output_tokens = torch.tensor(output_tokens, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
                output_scores=torch.full(output_tokens.size(), 1.0)
            elif self.args.decode_direction == "backward":
                links = r_links
                if self.args.decode_strategy == "lookahead":
                    output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                    links_idx = (links + unreduced_logits.unsqueeze(1) * self.args.decode_beta).max(dim=-1)[1].cpu().tolist() # batch * prelen
                elif self.args.decode_strategy == "greedy":
                    output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                    links_idx = links.max(dim=-1)[1].cpu().tolist() # batch * prelen

                unpad_output_tokens = []
                for i, length in enumerate(output_length):
                    last = unreduced_tokens[i][length-1]
                    j = length-1
                    res = [last]
                    while j != 0:
                        j = links_idx[i][j]
                        now_token = unreduced_tokens[i][j]
                        if now_token != self.tgt_dict.pad_index and now_token != last:
                            res.insert(0,now_token)
                        last = now_token
                    unpad_output_tokens.append(res)
                output_seqlen = max([len(res) for res in unpad_output_tokens])
                output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in unpad_output_tokens]
                output_tokens = torch.tensor(output_tokens, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
                output_scores=torch.full(output_tokens.size(), 1.0)
            elif self.args.decode_direction == "bidirection":
                #Perform Forward Decoding, Calculate the confidence of each sample
                links = n_links
                if self.args.decode_strategy == "lookahead":
                    output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                    links_idx = (links + unreduced_logits.unsqueeze(1) * self.args.decode_beta).max(dim=-1)[1].cpu().tolist() # batch * prelen
                    links_logits = (links + unreduced_logits.unsqueeze(1) * self.args.decode_beta).max(dim=-1)[0]
                elif self.args.decode_strategy == "greedy":
                    output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                    links_idx = links.max(dim=-1)[1].cpu().tolist() # batch * prelen
                    links_logits = links.max(dim=-1)[0]
                unpad_output_tokens = []
                n_logits = [] 
                for i, length in enumerate(output_length):
                    last = unreduced_tokens[i][0]
                    j = 0
                    res = [last]
                    logit = 0
                    logit_num = 0
                    while j != length - 1:
                        j = links_idx[i][j]
                        now_token = unreduced_tokens[i][j]
                        if now_token != self.tgt_dict.pad_index and now_token != last:
                            res.append(now_token)
                            if links_logits[i][j] != float('-inf'):
                                logit += links_logits[i][j]
                                logit_num += 1
                        last = now_token
                    if logit_num != 0:
                        logit /= logit_num
                    else:
                        logit = float('-inf')
                    unpad_output_tokens.append(res)
                    n_logits.append(logit)

                output_seqlen = max([len(res) for res in unpad_output_tokens])
                _output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in unpad_output_tokens]
                output_tokens_forward = torch.tensor(_output_tokens, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
                #Perform Backward Decoding
                links = r_links
                if self.args.decode_strategy == "lookahead":
                    output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                    links_idx = (links + unreduced_logits.unsqueeze(1) * self.args.decode_beta).max(dim=-1)[1].cpu().tolist() # batch * prelen
                    links_logits = (links + unreduced_logits.unsqueeze(1) * self.args.decode_beta).max(dim=-1)[0]
                elif self.args.decode_strategy == "greedy":
                    output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                    links_idx = links.max(dim=-1)[1].cpu().tolist() # batch * prelen
                    links_logits = links.max(dim=-1)[0]
                unpad_output_tokens = []
                r_logits = []
                for i, length in enumerate(output_length):
                    last = unreduced_tokens[i][length-1]
                    j = length-1
                    logit = 0
                    res = [last]
                    logit_num = 0
                    while j != 0:
                        j = links_idx[i][j]
                        now_token = unreduced_tokens[i][j]
                        if now_token != self.tgt_dict.pad_index and now_token != last:
                            res.insert(0,now_token)
                            if links_logits[i][j] != float('-inf'):
                                logit += links_logits[i][j]
                                logit_num += 1
                        last = now_token
                    unpad_output_tokens.append(res)
                    if logit_num != 0:
                        logit /= logit_num
                    else:
                        logit = float('-inf')
                    r_logits.append(logit)
                output_seqlen = max([len(res) for res in unpad_output_tokens])
                output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in unpad_output_tokens]
                output_tokens_backward = torch.tensor(output_tokens, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
                unpad_output_tokens,output_scores = self._bidir_argmax_selection(n_logits,r_logits,output_tokens_forward,output_tokens_backward)
                output_seqlen = max([len(res) for res in unpad_output_tokens])
                output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in unpad_output_tokens]
                output_tokens = torch.tensor(output_tokens, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
                output_scores=torch.full(output_tokens.size(), 1.0)
        elif self.args.decode_strategy in ["viterbi", "jointviterbi"]:
            if self.args.decode_direction == "forward":
                #print('not implemented')
                #exit()
                links =  n_links
                scores = []
                indexs = []
                # batch * graph_length
                alpha_t = links[:,0]
                if self.args.decode_strategy == "jointviterbi":
                    alpha_t += unreduced_logits[:,0].unsqueeze(1)
                batch_size, graph_length, _ = links.size()
                alpha_t += unreduced_logits
                scores.append(alpha_t)
                
                # the exact max_length should be graph_length - 2, but we can reduce it to an appropriate extent to speedup decoding
                max_length = int(2 * graph_length / self.args.src_upsample_scale)
                for i in range(max_length - 1):
                    alpha_t, index = torch.max(alpha_t.unsqueeze(-1) + links, dim = 1)
                    if self.args.decode_strategy == "jointviterbi":
                        alpha_t += unreduced_logits
                    scores.append(alpha_t)
                    indexs.append(index)

                # max_length * batch * graph_length
                indexs = torch.stack(indexs, dim = 0)
                scores = torch.stack(scores, dim = 0)
                link_last = torch.gather(links, -1, (output_length - 1).view(batch_size, 1, 1).repeat(1, graph_length, 1)).view(1, batch_size, graph_length)
                scores += link_last

                # max_length * batch
                scores, max_idx = torch.max(scores, dim = -1)
                lengths = torch.arange(max_length).unsqueeze(-1).repeat(1, batch_size) + 1
                length_penalty = (lengths ** self.args.decode_viterbibeta).cuda(scores.get_device())
                scores = scores / length_penalty
                max_score, pred_length = torch.max(scores, dim = 0)
                pred_length = pred_length + 1

                initial_idx = torch.gather(max_idx, 0, (pred_length - 1).view(1, batch_size)).view(batch_size).tolist()
                unpad_output_tokens = []
                indexs = indexs.tolist()
                pred_length = pred_length.tolist()
                for i, length in enumerate(pred_length):
                    j = initial_idx[i]
                    last = unreduced_tokens[i][j]
                    res = [last]
                    for k in range(length - 1):
                        j = indexs[length - k - 2][i][j]
                        now_token = unreduced_tokens[i][j]
                        if now_token != self.tgt_dict.pad_index and now_token != last:
                            res.insert(0, now_token)
                        last = now_token
                    unpad_output_tokens.append(res)

                output_seqlen = max([len(res) for res in unpad_output_tokens])
                output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in unpad_output_tokens]
                output_tokens = torch.tensor(output_tokens, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
            else:
                print('not implemented')
                exit()
        elif self.args.decode_strategy in ["hybrid_argmax_greedy", "hybrid_argmax_lookahead"]:
            argmax_tokens = torch.argmax(unreduced_logits.masked_fill(~valid_tgt_pos_mask,float("-inf")).masked_fill(~token_positions,float("-inf")),dim=1)
            #Caluculate Valid Position Masks
            if self.args.decode_strategy == "hybrid_argmax_lookahead":
                output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                n_links_idx = (n_links + unreduced_logits.unsqueeze(1) * self.args.decode_beta).max(dim=-1)[1].cpu().tolist() # batch * prelen
                r_links_idx = (r_links + unreduced_logits.unsqueeze(1) * self.args.decode_beta).max(dim=-1)[1].cpu().tolist() # batch * prelen
            elif self.args.decode_strategy == "hybrid_argmax_greedy":
                #print(n_links.shape)
                output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                n_links_idx = n_links.max(dim=-1)[1].cpu().tolist() # batch * prelen
                r_links_idx = r_links.max(dim=-1)[1].cpu().tolist() # batch * prelen
            unpad_output_tokens = []
            #Search from the argmax token to right
            #view_token_index = 2
            for i, length in enumerate(output_length):
                last = unreduced_tokens[i][argmax_tokens[i]]
                j = argmax_tokens[i]
                res = [last]
                #if i == view_token_index:
                    #print(res)
                while j != length - 1:
                    j = n_links_idx[i][j]
                    now_token = unreduced_tokens[i][j]
                    if now_token != self.tgt_dict.pad_index and now_token != last:
                        res.append(now_token)
                    last = now_token
                    #if i == view_token_index:
                        #print(res)
                unpad_output_tokens.append(res)
            #Search from the argmax token to left
            for i, length in enumerate(output_length):
                last = unreduced_tokens[i][argmax_tokens[i]]
                j = argmax_tokens[i]
                res = unpad_output_tokens[i]
                while j != 0:
                    j = r_links_idx[i][j]
                    now_token = unreduced_tokens[i][j]
                    if now_token != self.tgt_dict.pad_index and now_token != last:
                        res.insert(0,now_token)
                    #if i == view_token_index:
                        #print(res)
                    last = now_token
            #exit()
            output_seqlen = max([len(res) for res in unpad_output_tokens])
            output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in unpad_output_tokens]
            output_tokens = torch.tensor(output_tokens, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
            output_scores=torch.full(output_tokens.size(), 1.0)
        elif self.args.decode_strategy == "beamsearch":
            if self.args.decode_direction == "forward":
                links = n_links
                batch_size, prelen, _ = links.shape
                #link: b * pre * pre
                #output_logits: b * pre * dict
                assert batch_size <= self.args.decode_max_batchsize, "Please set --decode-max-batchsize for beamsearch with a larger batch size"
                top_logits, top_logits_idx = output_logits.log_softmax(dim=-1).topk(self.args.decode_top_cand_n, dim=-1)
                #top_logits: logits of top five tokens
                #top_logits_idx: logit index of top five tokens
                dagscores_arr = (links.unsqueeze(-1) + top_logits.unsqueeze(1) * self.args.decode_beta)  # batch * prelen * prelen * top_cand_n
                dagscores, top_cand_idx = dagscores_arr.reshape(batch_size, prelen, -1).topk(self.args.decode_top_cand_n, dim=-1) # batch * prelen * top_cand_n
                #reshape: 合并�?路径和token的概率，也就是说从一个位置出发，下一个可能的位置(prelen) * 选择下一个可能的tokens(top k)�?概率
                nextstep_idx = torch.div(top_cand_idx, self.args.decode_top_cand_n, rounding_mode="floor") # batch * prelen * top_cand_n
                #这里可以得到下一个step的index和score�?b * pre * candidate_num)
                #也就是说他每个位置只往后跳topn个可能，他这个搜索还是一个简化之后的搜索啊�?                
                logits_idx_idx = top_cand_idx % self.args.decode_top_cand_n # batch * prelen * top_cand_n
                #这个意思是说每一个位置往后跳到第几个candidate(selected from top k)
                idx1 = torch.arange(batch_size, device=links.device).unsqueeze(-1).unsqueeze(-1).expand(*nextstep_idx.shape)
                logits_idx = top_logits_idx[idx1, nextstep_idx, logits_idx_idx] # batch * prelen * top_cand_n
                #这里可以拿到每个topk token的logits
                rearange_idx = logits_idx.sort(dim=-1)[1]
                #这个idx其实�?-5之间排序的indices
                dagscores = dagscores.gather(-1, rearange_idx) # batch * prelen * top_cand_n
                nextstep_idx = nextstep_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n
                logits_idx = logits_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n
                #这里是按照概率重新排序好�?                
                dagscores = np.ascontiguousarray(dagscores.cpu().numpy())
                nextstep_idx = np.ascontiguousarray(nextstep_idx.int().cpu().numpy())
                logits_idx = np.ascontiguousarray(logits_idx.int().cpu().numpy())
                output_length_cpu = np.ascontiguousarray(output_length.int().cpu().numpy())
                #也就是说前面应该是把每一个step的top-k tokens * edges的概率算出来，排好序，然后交给c去计算的�?                
                res, score = self.dag_search.dag_search(dagscores, nextstep_idx, logits_idx,
                    output_length_cpu,
                    self.args.decode_alpha,
                    self.args.decode_gamma,
                    self.args.decode_beamsize,
                    self.args.decode_max_beam_per_length,
                    self.args.decode_top_p,
                    self.tgt_dict.pad_index,
                    self.tgt_dict.bos_index,
                    1 if self.args.decode_dedup else 0
                )
                output_tokens = torch.tensor(res, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
                output_scores = torch.tensor(score, device=decoder_out.output_scores.device, dtype=decoder_out.output_scores.dtype).unsqueeze(dim=-1).expand(*output_tokens.shape)
                #print(output_tokens.shape)
                #print(output_scores.shape)
                '''for item in output_tokens:
                    print(item)
                for item in output_scores:
                    print(item)
                exit()'''
            elif self.args.decode_direction == "backward":
                links = r_links
                batch_size, prelen, _ = links.shape
                #b * prelen * prelen
                '''print(len(links))
                for link in links:
                    print('--------------------------------------new link--------------------------------------------------')
                    for item in link:
                        print(item)'''
                #print(links.shape)
                #exit()

                assert batch_size <= self.args.decode_max_batchsize, "Please set --decode-max-batchsize for beamsearch with a larger batch size"

                top_logits, top_logits_idx = output_logits.log_softmax(dim=-1).topk(self.args.decode_top_cand_n, dim=-1)
                #b * prelen * topk
                dagscores_arr = (links.unsqueeze(-1) + top_logits.unsqueeze(1) * self.args.decode_beta)  # batch * prelen * prelen * top_cand_n
                dagscores, top_cand_idx = dagscores_arr.reshape(batch_size, prelen, -1).topk(self.args.decode_top_cand_n, dim=-1) # batch * prelen * top_cand_n
                nextstep_idx = torch.div(top_cand_idx, self.args.decode_top_cand_n, rounding_mode="floor") # batch * prelen * top_cand_n
                logits_idx_idx = top_cand_idx % self.args.decode_top_cand_n # batch * prelen * top_cand_n
                idx1 = torch.arange(batch_size, device=links.device).unsqueeze(-1).unsqueeze(-1).expand(*nextstep_idx.shape)
                logits_idx = top_logits_idx[idx1, nextstep_idx, logits_idx_idx] # batch * prelen * top_cand_n

                rearange_idx = logits_idx.sort(dim=-1)[1]
                dagscores = dagscores.gather(-1, rearange_idx) # batch * prelen * top_cand_n
                nextstep_idx = nextstep_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n
                logits_idx = logits_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n
                dagscores,nextstep_idx,logits_idx = self._reverse_search(dagscores,nextstep_idx,logits_idx,tgt_length=output_length)
                dagscores = np.ascontiguousarray(dagscores.cpu().numpy())
                nextstep_idx = np.ascontiguousarray(nextstep_idx.int().cpu().numpy())
                logits_idx = np.ascontiguousarray(logits_idx.int().cpu().numpy())
                output_length_cpu = np.ascontiguousarray(output_length.int().cpu().numpy())
                #其实只要把所有index相关的items反过来塞进去就行了�?                #好，下一个问题是idx也要反过来才行�?                
                res, score = self.dag_search.dag_search(dagscores, nextstep_idx, logits_idx,
                    output_length_cpu,
                    self.args.decode_alpha,
                    self.args.decode_gamma,
                    self.args.decode_beamsize,
                    self.args.decode_max_beam_per_length,
                    self.args.decode_top_p,
                    self.tgt_dict.pad_index,
                    self.tgt_dict.bos_index,
                    1 if self.args.decode_dedup else 0
                )
                output_tokens = torch.tensor(res, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
                output_scores = torch.tensor(score, device=decoder_out.output_scores.device, dtype=decoder_out.output_scores.dtype).unsqueeze(dim=-1).expand(*output_tokens.shape)
                output_tokens,output_scores = self._reverse_search_output(output_tokens,output_scores)
            elif self.args.decode_direction == "bidirection":
                #Perform Forward & Backward Beam Search, Select the most probable samples.
                #注意，这两个结果经过了padding，但还是要重新padding一下才可以�?                #Forward Beam First
                links = n_links
                batch_size, prelen, _ = links.shape
                #link: b * pre * pre
                #output_logits: b * pre * dict
                assert batch_size <= self.args.decode_max_batchsize, "Please set --decode-max-batchsize for beamsearch with a larger batch size"
                top_logits, top_logits_idx = output_logits.log_softmax(dim=-1).topk(self.args.decode_top_cand_n, dim=-1)
                #top_logits: logits of top five tokens
                #top_logits_idx: logit index of top five tokens
                dagscores_arr = (links.unsqueeze(-1) + top_logits.unsqueeze(1) * self.args.decode_beta)  # batch * prelen * prelen * top_cand_n
                dagscores, top_cand_idx = dagscores_arr.reshape(batch_size, prelen, -1).topk(self.args.decode_top_cand_n, dim=-1) # batch * prelen * top_cand_n
                #reshape: 合并�?路径和token的概率，也就是说从一个位置出发，下一个可能的位置(prelen) * 选择下一个可能的tokens(top k)�?概率
                nextstep_idx = torch.div(top_cand_idx, self.args.decode_top_cand_n, rounding_mode="floor") # batch * prelen * top_cand_n
                #这里可以得到下一个step的index和score�?b * pre * candidate_num)
                #也就是说他每个位置只往后跳topn个可能，他这个搜索还是一个简化之后的搜索啊�?                
                logits_idx_idx = top_cand_idx % self.args.decode_top_cand_n # batch * prelen * top_cand_n
                #这个意思是说每一个位置往后跳到第几个candidate(selected from top k)
                idx1 = torch.arange(batch_size, device=links.device).unsqueeze(-1).unsqueeze(-1).expand(*nextstep_idx.shape)
                logits_idx = top_logits_idx[idx1, nextstep_idx, logits_idx_idx] # batch * prelen * top_cand_n
                #这里可以拿到每个topk token的logits
                rearange_idx = logits_idx.sort(dim=-1)[1]
                #这个idx其实�?-5之间排序的indices
                dagscores = dagscores.gather(-1, rearange_idx) # batch * prelen * top_cand_n
                nextstep_idx = nextstep_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n
                logits_idx = logits_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n
                #这里是按照概率重新排序好�?                
                dagscores = np.ascontiguousarray(dagscores.cpu().numpy())
                nextstep_idx = np.ascontiguousarray(nextstep_idx.int().cpu().numpy())
                logits_idx = np.ascontiguousarray(logits_idx.int().cpu().numpy())
                output_length_cpu = np.ascontiguousarray(output_length.int().cpu().numpy())
                #也就是说前面应该是把每一个step的top-k tokens * edges的概率算出来，排好序，然后交给c去计算的�?                
                res, score = self.dag_search.dag_search(dagscores, nextstep_idx, logits_idx,
                    output_length_cpu,
                    self.args.decode_alpha,
                    self.args.decode_gamma,
                    self.args.decode_beamsize,
                    self.args.decode_max_beam_per_length,
                    self.args.decode_top_p,
                    self.tgt_dict.pad_index,
                    self.tgt_dict.bos_index,
                    1 if self.args.decode_dedup else 0
                )
                n_output_tokens = torch.tensor(res, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
                n_output_scores = torch.tensor(score, device=decoder_out.output_scores.device, dtype=decoder_out.output_scores.dtype).unsqueeze(dim=-1).expand(*output_tokens.shape)
                #Then, Backward Beam
                links = r_links
                batch_size, prelen, _ = links.shape
                assert batch_size <= self.args.decode_max_batchsize, "Please set --decode-max-batchsize for beamsearch with a larger batch size"

                top_logits, top_logits_idx = output_logits.log_softmax(dim=-1).topk(self.args.decode_top_cand_n, dim=-1)
                #b * prelen * topk
                dagscores_arr = (links.unsqueeze(-1) + top_logits.unsqueeze(1) * self.args.decode_beta)  # batch * prelen * prelen * top_cand_n
                dagscores, top_cand_idx = dagscores_arr.reshape(batch_size, prelen, -1).topk(self.args.decode_top_cand_n, dim=-1) # batch * prelen * top_cand_n
                nextstep_idx = torch.div(top_cand_idx, self.args.decode_top_cand_n, rounding_mode="floor") # batch * prelen * top_cand_n
                logits_idx_idx = top_cand_idx % self.args.decode_top_cand_n # batch * prelen * top_cand_n
                idx1 = torch.arange(batch_size, device=links.device).unsqueeze(-1).unsqueeze(-1).expand(*nextstep_idx.shape)
                logits_idx = top_logits_idx[idx1, nextstep_idx, logits_idx_idx] # batch * prelen * top_cand_n

                rearange_idx = logits_idx.sort(dim=-1)[1]
                dagscores = dagscores.gather(-1, rearange_idx) # batch * prelen * top_cand_n
                nextstep_idx = nextstep_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n
                logits_idx = logits_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n
                dagscores,nextstep_idx,logits_idx = self._reverse_search(dagscores,nextstep_idx,logits_idx,tgt_length=output_length)
                dagscores = np.ascontiguousarray(dagscores.cpu().numpy())
                nextstep_idx = np.ascontiguousarray(nextstep_idx.int().cpu().numpy())
                logits_idx = np.ascontiguousarray(logits_idx.int().cpu().numpy())
                output_length_cpu = np.ascontiguousarray(output_length.int().cpu().numpy())
                #其实只要把所有index相关的items反过来塞进去就行了�?                #好，下一个问题是idx也要反过来才行�?                
                res, score = self.dag_search.dag_search(dagscores, nextstep_idx, logits_idx,
                    output_length_cpu,
                    self.args.decode_alpha,
                    self.args.decode_gamma,
                    self.args.decode_beamsize,
                    self.args.decode_max_beam_per_length,
                    self.args.decode_top_p,
                    self.tgt_dict.pad_index,
                    self.tgt_dict.bos_index,
                    1 if self.args.decode_dedup else 0
                )
                r_output_tokens = torch.tensor(res, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
                r_output_scores = torch.tensor(score, device=decoder_out.output_scores.device, dtype=decoder_out.output_scores.dtype).unsqueeze(dim=-1).expand(*output_tokens.shape)
                r_output_tokens,r_output_scores = self._reverse_search_output(r_output_tokens,r_output_scores)
                unpad_output_tokens,output_scores = self._bidir_argmax_selection(n_output_scores,r_output_scores,n_output_tokens,r_output_tokens,mode='beam')
                output_seqlen = max([len(res) for res in unpad_output_tokens])
                output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in unpad_output_tokens]
                output_tokens = torch.tensor(output_tokens, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
                output_scores=torch.full(output_tokens.size(), 1.0)
            elif self.args.decode_direction == "bidirection_select":
                #Perform Forward & Backward Beam Search, Select the most probable samples.
                #注意，这两个结果经过了padding，但还是要重新padding一下才可以�?                #Forward Beam First
                links = n_links
                batch_size, prelen, _ = links.shape
                #link: b * pre * pre
                #output_logits: b * pre * dict
                assert batch_size <= self.args.decode_max_batchsize, "Please set --decode-max-batchsize for beamsearch with a larger batch size"
                top_logits, top_logits_idx = output_logits.log_softmax(dim=-1).topk(self.args.decode_top_cand_n, dim=-1)
                #top_logits: logits of top five tokens
                #top_logits_idx: logit index of top five tokens
                dagscores_arr = (links.unsqueeze(-1) + top_logits.unsqueeze(1) * self.args.decode_beta)  # batch * prelen * prelen * top_cand_n
                dagscores, top_cand_idx = dagscores_arr.reshape(batch_size, prelen, -1).topk(self.args.decode_top_cand_n, dim=-1) # batch * prelen * top_cand_n
                #reshape: 合并�?路径和token的概率，也就是说从一个位置出发，下一个可能的位置(prelen) * 选择下一个可能的tokens(top k)�?概率
                nextstep_idx = torch.div(top_cand_idx, self.args.decode_top_cand_n, rounding_mode="floor") # batch * prelen * top_cand_n
                #这里可以得到下一个step的index和score�?b * pre * candidate_num)
                #也就是说他每个位置只往后跳topn个可能，他这个搜索还是一个简化之后的搜索啊�?                
                logits_idx_idx = top_cand_idx % self.args.decode_top_cand_n # batch * prelen * top_cand_n
                #这个意思是说每一个位置往后跳到第几个candidate(selected from top k)
                idx1 = torch.arange(batch_size, device=links.device).unsqueeze(-1).unsqueeze(-1).expand(*nextstep_idx.shape)
                logits_idx = top_logits_idx[idx1, nextstep_idx, logits_idx_idx] # batch * prelen * top_cand_n
                #这里可以拿到每个topk token的logits
                rearange_idx = logits_idx.sort(dim=-1)[1]
                #这个idx其实�?-5之间排序的indices
                dagscores = dagscores.gather(-1, rearange_idx) # batch * prelen * top_cand_n
                nextstep_idx = nextstep_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n
                logits_idx = logits_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n
                #这里是按照概率重新排序好�?                
                dagscores = np.ascontiguousarray(dagscores.cpu().numpy())
                nextstep_idx = np.ascontiguousarray(nextstep_idx.int().cpu().numpy())
                logits_idx = np.ascontiguousarray(logits_idx.int().cpu().numpy())
                output_length_cpu = np.ascontiguousarray(output_length.int().cpu().numpy())
                #也就是说前面应该是把每一个step的top-k tokens * edges的概率算出来，排好序，然后交给c去计算的�?                
                res, score = self.dag_search.dag_search(dagscores, nextstep_idx, logits_idx,
                    output_length_cpu,
                    self.args.decode_alpha,
                    self.args.decode_gamma,
                    self.args.decode_beamsize,
                    self.args.decode_max_beam_per_length,
                    self.args.decode_top_p,
                    self.tgt_dict.pad_index,
                    self.tgt_dict.bos_index,
                    1 if self.args.decode_dedup else 0
                )
                n_output_tokens = torch.tensor(res, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
                n_output_scores = torch.tensor(score, device=decoder_out.output_scores.device, dtype=decoder_out.output_scores.dtype).unsqueeze(dim=-1).expand(*output_tokens.shape)
                #Then, Backward Beam
                links = r_links
                batch_size, prelen, _ = links.shape
                assert batch_size <= self.args.decode_max_batchsize, "Please set --decode-max-batchsize for beamsearch with a larger batch size"

                top_logits, top_logits_idx = output_logits.log_softmax(dim=-1).topk(self.args.decode_top_cand_n, dim=-1)
                #b * prelen * topk
                dagscores_arr = (links.unsqueeze(-1) + top_logits.unsqueeze(1) * self.args.decode_beta)  # batch * prelen * prelen * top_cand_n
                dagscores, top_cand_idx = dagscores_arr.reshape(batch_size, prelen, -1).topk(self.args.decode_top_cand_n, dim=-1) # batch * prelen * top_cand_n
                nextstep_idx = torch.div(top_cand_idx, self.args.decode_top_cand_n, rounding_mode="floor") # batch * prelen * top_cand_n
                logits_idx_idx = top_cand_idx % self.args.decode_top_cand_n # batch * prelen * top_cand_n
                idx1 = torch.arange(batch_size, device=links.device).unsqueeze(-1).unsqueeze(-1).expand(*nextstep_idx.shape)
                logits_idx = top_logits_idx[idx1, nextstep_idx, logits_idx_idx] # batch * prelen * top_cand_n

                rearange_idx = logits_idx.sort(dim=-1)[1]
                dagscores = dagscores.gather(-1, rearange_idx) # batch * prelen * top_cand_n
                nextstep_idx = nextstep_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n
                logits_idx = logits_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n
                dagscores,nextstep_idx,logits_idx = self._reverse_search(dagscores,nextstep_idx,logits_idx,tgt_length=output_length)
                dagscores = np.ascontiguousarray(dagscores.cpu().numpy())
                nextstep_idx = np.ascontiguousarray(nextstep_idx.int().cpu().numpy())
                logits_idx = np.ascontiguousarray(logits_idx.int().cpu().numpy())
                output_length_cpu = np.ascontiguousarray(output_length.int().cpu().numpy())
                #其实只要把所有index相关的items反过来塞进去就行了�?                #好，下一个问题是idx也要反过来才行�?                
                res, score = self.dag_search.dag_search(dagscores, nextstep_idx, logits_idx,
                    output_length_cpu,
                    self.args.decode_alpha,
                    self.args.decode_gamma,
                    self.args.decode_beamsize,
                    self.args.decode_max_beam_per_length,
                    self.args.decode_top_p,
                    self.tgt_dict.pad_index,
                    self.tgt_dict.bos_index,
                    1 if self.args.decode_dedup else 0
                )
                r_output_tokens = torch.tensor(res, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
                r_output_scores = torch.tensor(score, device=decoder_out.output_scores.device, dtype=decoder_out.output_scores.dtype).unsqueeze(dim=-1).expand(*output_tokens.shape)
                r_output_tokens,r_output_scores = self._reverse_search_output(r_output_tokens,r_output_scores)
                bi_dp_selection_pack = {
                    'output_logits': output_logits,
                    'n_links': n_links,
                    'r_links': r_links,
                    'output_masks': output_tokens.ne(self.tgt_dict.pad_index),
                    'output_length':output_length,
                    'n_tgt_mask': n_output_tokens.ne(self.pad),
                    'r_tgt_mask': r_output_tokens.ne(self.pad),
                }
                unpad_output_tokens,output_scores = self._bidir_argmax_selection(n_output_scores,r_output_scores,n_output_tokens,r_output_tokens,mode='bi-select',bi_select_pack= bi_dp_selection_pack)
                output_seqlen = max([len(res) for res in unpad_output_tokens])
                output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in unpad_output_tokens]
                output_tokens = torch.tensor(output_tokens, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
                output_scores=torch.full(output_tokens.size(), 1.0)
        elif self.args.decode_strategy in ["hybrid_argmax_beam"]:
            #Acquire the argmax topk tokens
            argmax_token_logits,argmax_tokens = torch.topk(unreduced_logits.masked_fill(~valid_tgt_pos_mask,float("-inf")).masked_fill(~token_positions,float("-inf")),dim=1,k=self.args.argmax_token_num)
            #print(unreduced_logits.masked_fill(~valid_tgt_pos_mask,float("-inf")).masked_fill(~token_positions,float("-inf")))
            #print(argmax_token_logits)
            #print(argmax_tokens)
            unreduced_tokens = torch.tensor(unreduced_tokens,device=argmax_tokens.device)
            argmax_token_idx = argmax_tokens.clone()
            for i in range(bsz):
                argmax_token_idx[i][:] = unreduced_tokens[i][argmax_tokens[i]]
                #print(unreduced_tokens[i][argmax_tokens[i]])
            #Perform Forward & Backward Beam Search
            #Initiate Forward & Backward DAG Score
            batch_size, prelen, _ = n_links.shape
            #link: b * pre * pre
            #output_logits: b * pre * dict
            assert batch_size <= self.args.decode_max_batchsize, "Please set --decode-max-batchsize for beamsearch with a larger batch size"
            top_logits, top_logits_idx = output_logits.log_softmax(dim=-1).topk(self.args.decode_top_cand_n, dim=-1)
            #top_logits: logits of top five tokens b * pre * tok
            #top_logits_idx: logit index of top five tokens
            #print('N-Link & R_Link Shape')
            #print(n_links.unsqueeze(-1).shape,r_links.unsqueeze(-1).shape)
            #print(top_logits.shape)#b * prelen * k
            #print(top_logits.unsqueeze(1).shape)#b * 1 * prelen * k
            n_dagscores_arr = (n_links.unsqueeze(-1) + top_logits.unsqueeze(1) * self.args.decode_beta)  # batch * prelen * prelen * top_cand_n
            #print(n_dagscores_arr.shape)
            n_dagscores, n_top_cand_idx = n_dagscores_arr.reshape(batch_size, prelen, -1).topk(self.args.decode_top_cand_n, dim=-1) # batch * prelen * (prelen*top_cand_n)
            r_dagscores_arr = (r_links.unsqueeze(-1) + top_logits.unsqueeze(1) * self.args.decode_beta)  # batch * prelen * prelen * top_cand_n
            r_dagscores, r_top_cand_idx = r_dagscores_arr.reshape(batch_size, prelen, -1).topk(self.args.decode_top_cand_n, dim=-1) # batch * prelen * top_cand_n
            #reshape: 合并�?路径和token的概率，也就是说从一个位置出发，下一个可能的位置(prelen) * 选择下一个可能的tokens(top k)�?概率
            n_nextstep_idx = torch.div(n_top_cand_idx, self.args.decode_top_cand_n, rounding_mode="floor") # batch * prelen * top_cand_n
            r_nextstep_idx = torch.div(r_top_cand_idx, self.args.decode_top_cand_n, rounding_mode="floor") # batch * prelen * top_cand_n
            #这里可以得到下一个step的index和score�?b * pre * candidate_num)
            #也就是说他每个位置只往后跳topn个可能，他这个搜索还是一个简化之后的搜索啊�?            
            n_logits_idx_idx = n_top_cand_idx % self.args.decode_top_cand_n # batch * prelen * top_cand_n
            r_logits_idx_idx = r_top_cand_idx % self.args.decode_top_cand_n # batch * prelen * top_cand_n
            #这个意思是说每一个位置往后跳到第几个candidate(selected from top k)
            n_idx1 = torch.arange(batch_size, device=n_links.device).unsqueeze(-1).unsqueeze(-1).expand(*n_nextstep_idx.shape)
            n_logits_idx = top_logits_idx[n_idx1, n_nextstep_idx, n_logits_idx_idx] # batch * prelen * top_cand_n
            r_idx1 = torch.arange(batch_size, device=r_links.device).unsqueeze(-1).unsqueeze(-1).expand(*r_nextstep_idx.shape)
            r_logits_idx = top_logits_idx[r_idx1, r_nextstep_idx, r_logits_idx_idx] # batch * prelen * top_cand_n
            #这里可以拿到每个topk token的logits
            n_rearange_idx = n_logits_idx.sort(dim=-1)[1]
            r_rearange_idx = r_logits_idx.sort(dim=-1)[1]
            #这个idx其实�?-5之间排序的indices
            n_dagscores = n_dagscores.gather(-1, n_rearange_idx) # batch * prelen * top_cand_n
            n_nextstep_idx = n_nextstep_idx.gather(-1, n_rearange_idx) # batch * prelen * top_cand_n
            n_logits_idx = n_logits_idx.gather(-1, n_rearange_idx) # batch * prelen * top_cand_n
            r_dagscores = r_dagscores.gather(-1, r_rearange_idx) # batch * prelen * top_cand_n
            r_nextstep_idx = r_nextstep_idx.gather(-1, r_rearange_idx) # batch * prelen * top_cand_n
            r_logits_idx = r_logits_idx.gather(-1, r_rearange_idx) # batch * prelen * top_cand_n
            #print('batch size')
            #print(n_links.shape)
            res,score = self._argmax_beam_samples( n_dagscores,r_dagscores, n_nextstep_idx, r_nextstep_idx, n_logits_idx, r_logits_idx, output_length, argmax_tokens,argmax_token_idx,argmax_token_logits)
            #这里是按照概率重新排序好�?            #Aggregate the Search Result
            #Return the tokens and confidence
            #a = 0
            #print(len(score[0]))
            output_tokens = torch.tensor(res, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
            output_scores = torch.tensor(score, device=decoder_out.output_scores.device, dtype=decoder_out.output_scores.dtype).unsqueeze(dim=-1).expand(*output_tokens.shape)
        if history is not None:
            history.append(output_tokens.clone())
        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

class BiGlatLinkDecoder(NATransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.init_link_feature(args)

    def init_link_feature(self, args):
        links_feature = self.args.links_feature.split(":")
        links_dim = 0
        if "feature" in links_feature:
            links_dim += args.decoder_embed_dim
        if "position" in links_feature:
            self.link_positional = PositionalEmbedding(args.max_target_positions, args.decoder_embed_dim, self.padding_idx, True)
            links_dim += args.decoder_embed_dim
        elif "sinposition" in links_feature:
            self.link_positional = PositionalEmbedding(args.max_target_positions, args.decoder_embed_dim, self.padding_idx, False)
            links_dim += args.decoder_embed_dim
        else:
            self.link_positional = None
        if "share" in links_feature:
            self.n_link_query_linear = nn.Linear(links_dim, args.decoder_embed_dim)
            self.n_link_key_linear = nn.Linear(links_dim, args.decoder_embed_dim)
            self.n_link_gate_linear = nn.Linear(links_dim, args.decoder_attention_heads)
            self.r_link_query_linear = self.n_link_query_linear
            self.r_link_key_linear = self.n_link_key_linear
            self.r_link_gate_linear = self.n_link_gate_linear
        else:
            self.n_link_query_linear = nn.Linear(links_dim, args.decoder_embed_dim)
            self.n_link_key_linear = nn.Linear(links_dim, args.decoder_embed_dim)
            self.n_link_gate_linear = nn.Linear(links_dim, args.decoder_attention_heads)
            self.r_link_query_linear = nn.Linear(links_dim, args.decoder_embed_dim)
            self.r_link_key_linear = nn.Linear(links_dim, args.decoder_embed_dim)
            self.r_link_gate_linear = nn.Linear(links_dim, args.decoder_attention_heads)

    @staticmethod
    def add_args(parser):
        pass

@register_model_architecture(
    "bi_glat_decomposed_link", "bi_glat_decomposed_link_6e6d512"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)

@register_model_architecture(
    "bi_glat_decomposed_link", "bi_glat_decomposed_link_base"
)
def base_architecture2(args):
    base_architecture(args)
