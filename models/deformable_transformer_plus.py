# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from models.structures import Boxes, matched_boxlist_iou, pairwise_iou

from util.misc import inverse_sigmoid
from util.box_ops import box_cxcywh_to_xyxy
from models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, num_cross_attn_layers=6,
                 dim_feedforward=1024, dropout=0.1, num_ps_new_born=10,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, decoder_self_cross=True, sigmoid_attn=False,
                 extra_track_attn=False, memory_bank=False):
        super().__init__()

        self.new_frame_adaptor = None
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points,
                                                          sigmoid_attn=sigmoid_attn)
        self.encoder = DeformableTransformerEncoder(
            encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, decoder_self_cross,
                                                          sigmoid_attn=sigmoid_attn, extra_track_attn=extra_track_attn,
                                                          memory_bank=memory_bank)
        self.track_decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, return_intermediate_dec)
        self.detect_decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, return_intermediate_dec)
        
        cross_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.cross_attn_decoder = CrossAttentionDecoder(cross_decoder_layer, num_cross_attn_layers, return_intermediate=return_intermediate_dec)

        self.track_embedding = nn.Parameter(torch.Tensor(1, d_model))
        self.detect_embedding = nn.Parameter(torch.Tensor(1, d_model))
        self.reference_points = nn.Parameter(torch.Tensor(1, 4))
        self.return_intermediate_dec = return_intermediate_dec

        self.level_embed = nn.Parameter(
            torch.Tensor(num_feature_levels, d_model))
        
        self.new_born_embed = nn.Parameter(torch.Tensor(1, d_model))
        self.num_ps_new_born = num_ps_new_born

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)
        normal_(self.new_born_embed)
        normal_(self.track_embedding)
        normal_(self.detect_embedding)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(
                _cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat(
                [valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (
            output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(
            ~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, ref_pts=None, mem_bank=None, mem_bank_pad_mask=None, attn_mask=None, num_dts=0):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index,
                              valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        # if self.two_stage:
        #     output_memory, output_proposals = self.gen_encoder_output_proposals(
        #         memory, mask_flatten, spatial_shapes)

        #     # hack implementation for two-stage Deformable DETR
        #     enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](
        #         output_memory)
        #     enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](
        #         output_memory) + output_proposals

        #     topk = self.two_stage_num_proposals
        #     topk_proposals = torch.topk(
        #         enc_outputs_class[..., 0], topk, dim=1)[1]
        #     topk_coords_unact = torch.gather(
        #         enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        #     topk_coords_unact = topk_coords_unact.detach()
        #     reference_points = topk_coords_unact.sigmoid()
        #     init_reference_out = reference_points
        #     pos_trans_out = self.pos_trans_norm(self.pos_trans(
        #         self.get_proposal_pos_embed(topk_coords_unact)))
        #     query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        # else:
        tgt = query_embed.unsqueeze(0).expand(bs, -1, -1)
        reference_points = ref_pts.unsqueeze(0).expand(bs, -1, -1)
        init_reference_out = reference_points

        # # Update tgt by averaging the mem_bank
        # if mem_bank is not None:
        #     avg_mem_bank = mem_bank * (~mem_bank_pad_mask.unsqueeze(-1))
        #     avg_mem_bank = avg_mem_bank.sum(
        #         1) / (~mem_bank_pad_mask).sum(1).unsqueeze(-1)
        #     new_tgt = (tgt + avg_mem_bank) / 2.
        #     is_exist_mem = (~mem_bank_pad_mask).any(1)
        #     print(is_exist_mem.sum())
        #     tgt = torch.where(is_exist_mem.unsqueeze(-1), new_tgt, tgt)
        # decoder
        detect_tgt, detect_reference_points = tgt[:, :num_dts], reference_points[:, :num_dts]
        track_tgt, track_reference_points = tgt[:, num_dts:], reference_points[:, num_dts:]
        
        detect_tgt = torch.cat([self.detect_embedding.expand(detect_tgt.shape[0], 1, -1), detect_tgt], 1)
        track_tgt = torch.cat([self.track_embedding.expand(track_tgt.shape[0], 1, -1), track_tgt], 1)
        detect_reference_points = torch.cat([self.reference_points.expand(detect_reference_points.shape[0], 1, -1), detect_reference_points], 1)
        track_reference_points = torch.cat([self.reference_points.expand(track_reference_points.shape[0], 1, -1), track_reference_points], 1)
    


        hs_detect, inter_detect_references = self.detect_decoder(detect_tgt, detect_reference_points, memory,
                                                                    spatial_shapes, level_start_index,
                                                                    valid_ratios, mask_flatten,
                                                                    mem_bank, mem_bank_pad_mask, attn_mask)
        hs_track, inter_track_references = self.track_decoder(track_tgt, track_reference_points, memory,
                                                                spatial_shapes, level_start_index,
                                                                valid_ratios, mask_flatten,
                                                                mem_bank, mem_bank_pad_mask, attn_mask)
        
        if self.return_intermediate_dec:
            last_hs_detect = hs_detect[-1]
            last_hs_track = hs_track[-1]
            last_inter_detect_references = inter_detect_references[-1]
            last_inter_track_references = inter_track_references[-1]
        
        if self.num_ps_new_born > 0:
            last_hs_track_with_new_born = torch.cat([last_hs_track, self.new_born_embed.expand(last_hs_track.shape[0], self.num_ps_new_born, -1)], 1)
            last_inter_track_references_with_new_born = torch.cat([last_inter_track_references, torch.zeros(last_inter_track_references.shape[0], self.num_ps_new_born, last_inter_track_references.shape[-1], device=last_inter_track_references.device)], 1)
        else:    
            last_hs_track_with_new_born = last_hs_track
            last_inter_track_references_with_new_born = last_inter_track_references

        cross_hs_detect, cross_hs_track, cross_inter_detect_references, cross_inter_track_references = self.cross_attn_decoder(last_hs_detect, last_hs_track_with_new_born,
                                                                                                        tgt1_mask=None, tgt2_mask=None,
                                                                                                        tgt1_key_padding_mask=None, tgt2_key_padding_mask=None,
                                                                                                        tgt1_reference_points=last_inter_detect_references, 
                                                                                                        tgt2_reference_points=last_inter_track_references_with_new_born,
                                                                                                        num_tgt1=last_hs_detect.shape[1], num_tgt2=last_hs_track.shape[1],
                                                                                                        pos1=None, pos2=None)

        # Remove the first token
        if self.return_intermediate_dec:
            hs_detect = hs_detect[:,:, 1:]
            hs_track = hs_track[:,:, 1:]
            inter_detect_references = inter_detect_references[:,:, 1:]
            inter_track_references = inter_track_references[:,:, 1:]
            cross_hs_detect = cross_hs_detect[:,:, 1:]
            cross_hs_track = cross_hs_track[:,:, 1:hs_track.shape[2]]
            cross_inter_detect_references = cross_inter_detect_references[:,:, 1:]
            cross_inter_track_references = cross_inter_track_references[:,:, 1:hs_track.shape[2]]
        else:
            hs_detect = hs_detect[:, 1:]
            hs_track = hs_track[:, 1:]
            inter_detect_references = inter_detect_references[:, 1:]
            inter_track_references = inter_track_references[:, 1:]
            cross_hs_detect = cross_hs_detect[:, 1:]
            cross_hs_track = cross_hs_track[:, 1:]
            cross_inter_detect_references = cross_inter_detect_references[:, 1:]
            cross_inter_track_references = cross_inter_track_references[:, 1:]

        print(hs_detect.shape, hs_track.shape, inter_detect_references.shape, inter_track_references.shape, cross_hs_detect.shape, cross_hs_track.shape, cross_inter_detect_references.shape, cross_inter_track_references.shape)


        # hs, inter_references = self.track_decoder(tgt, reference_points, memory,
        #                                     spatial_shapes, level_start_index,
        #                                     valid_ratios, mask_flatten,
        #                                     mem_bank, mem_bank_pad_mask, attn_mask)

        # inter_references_out = inter_references
        # if self.two_stage:
        #     return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return {
            'hs_detect': hs_detect,
            'hs_track': hs_track,
            'inter_detect_references': inter_detect_references,
            'inter_track_references': inter_track_references,
            'cross_hs_detect': cross_hs_detect,
            'cross_hs_track': cross_hs_track,
            'cross_inter_detect_references': cross_inter_detect_references,
            'cross_inter_track_references': cross_inter_track_references,
            'init_detect_reference': init_reference_out[:, :num_dts],
            'init_track_reference': init_reference_out[:, num_dts:],
        }

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, sigmoid_attn=False):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(
            d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout_relu = ReLUDropout(dropout, True)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout_relu(self.linear1(src)))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(
            src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / \
                (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / \
                (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points,
                           spatial_shapes, level_start_index, padding_mask)

        return output


class ReLUDropout(torch.nn.Dropout):
    def forward(self, input):
        return relu_dropout(input, p=self.p, training=self.training, inplace=self.inplace)


def relu_dropout(x, p=0, inplace=False, training=False):
    if not training or p == 0:
        return x.clamp_(min=0) if inplace else x.clamp(min=0)

    mask = (x < 0) | (torch.rand_like(x) > 1 - p)
    return x.masked_fill_(mask, 0).div_(1 - p) if inplace else x.masked_fill(mask, 0).div(1 - p)


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, self_cross=True, sigmoid_attn=False,
                 extra_track_attn=False, memory_bank=False):
        super().__init__()

        self.self_cross = self_cross
        self.num_head = n_heads
        self.memory_bank = memory_bank

        # cross attention
        self.cross_attn = MSDeformAttn(
            d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout_relu = ReLUDropout(dropout, True)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # memory bank
        if self.memory_bank:
            self.temporal_attn = nn.MultiheadAttention(d_model, 8, dropout=0)
            self.temporal_fc1 = nn.Linear(d_model, d_ffn)
            self.temporal_fc2 = nn.Linear(d_ffn, d_model)
            self.temporal_norm1 = nn.LayerNorm(d_model)
            self.temporal_norm2 = nn.LayerNorm(d_model)

            position = torch.arange(5).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2)
                                 * (-math.log(10000.0) / d_model))
            pe = torch.zeros(5, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        # update track query_embed
        self.extra_track_attn = extra_track_attn
        if self.extra_track_attn:
            print('Training with Extra Self Attention in Every Decoder.', flush=True)
            self.update_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout)
            self.dropout5 = nn.Dropout(dropout)
            self.norm4 = nn.LayerNorm(d_model)

        if self_cross:
            print('Training with Self-Cross Attention.')
        else:
            print('Training with Cross-Self Attention.')

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout_relu(self.linear1(tgt)))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _forward_self_attn(self, tgt, query_pos, attn_mask=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        if attn_mask is not None:
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1),
                                  attn_mask=attn_mask)[0].transpose(0, 1)
        else:
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(
                0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        return self.norm2(tgt)

    def _forward_self_cross(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                            src_padding_mask=None, attn_mask=None):

        # self attention
        tgt = self._forward_self_attn(tgt, query_pos, attn_mask)
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

    def _forward_cross_self(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                            src_padding_mask=None, attn_mask=None):
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # self attention
        tgt = self._forward_self_attn(tgt, query_pos, attn_mask)
        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None, mem_bank=None, mem_bank_pad_mask=None, attn_mask=None):
        if self.self_cross:
            return self._forward_self_cross(tgt, query_pos, reference_points, src, src_spatial_shapes,
                                            level_start_index, src_padding_mask, attn_mask)
        return self._forward_cross_self(tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                                        src_padding_mask, attn_mask)


def pos2posemb(pos, num_pos_feats=64, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    posemb = pos[..., None] / dim_t
    posemb = torch.stack(
        (posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-3)
    return posemb


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                src_padding_mask=None, mem_bank=None, mem_bank_pad_mask=None, attn_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:,
                                                          :, None] * src_valid_ratios[:, None]
            query_pos = pos2posemb(reference_points)
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes,
                           src_level_start_index, src_padding_mask, mem_bank, mem_bank_pad_mask, attn_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + \
                        inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[...,
                                                        :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points
    

class CrossAttentionDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.tgt1_layers = _get_clones(decoder_layer, num_layers)
        self.tgt2_layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.box_embed = None

    def forward(self, tgt1, tgt2,
                tgt1_mask: Optional[Tensor] = None,
                tgt2_mask: Optional[Tensor] = None,
                tgt1_key_padding_mask: Optional[Tensor] = None,
                tgt2_key_padding_mask: Optional[Tensor] = None,
                tgt1_reference_points: Optional[Tensor] = None,
                tgt2_reference_points: Optional[Tensor] = None,
                num_tgt1: Optional[int] = None,
                num_tgt2: Optional[int] = None,
                pos1: Optional[Tensor] = None,
                pos2: Optional[Tensor] = None):

        output1 = tgt1
        output2 = tgt2

        intermediate1 = []
        intermediate2 = []

        intermediate1_reference_points = []
        intermediate2_reference_points = []

        for lid in range(self.num_layers):
            layer_tgt1 = self.tgt1_layers[lid]
            layer_tgt2 = self.tgt2_layers[lid]
            # Init pos embedding
            pe1 = pos2posemb(tgt1_reference_points)
            pe2 = pos2posemb(tgt2_reference_points)

            if pos1 is not None:
                cross_pos1 = pe1 + pos1
            else:
                cross_pos1 = pe1
            if pos2 is not None:
                cross_pos2 = pe2 + pos2         
            else:
                cross_pos2 = pe2

            new_output1 = layer_tgt1(output1[:, :num_tgt1].transpose(0, 1)
                                     , output2.transpose(0, 1), tgt1_mask, tgt2_mask,
                                    tgt1_key_padding_mask, tgt2_key_padding_mask, cross_pos2.transpose(0, 1), pe1[:, :num_tgt1].transpose(0, 1))
            new_output2 = layer_tgt2(output2[:, :num_tgt2].transpose(0, 1), output1.transpose(0, 1), tgt2_mask, tgt1_mask,
                                tgt2_key_padding_mask, tgt1_key_padding_mask, cross_pos1.transpose(0, 1), pe2[:, :num_tgt2].transpose(0, 1))
            
            output1[:, :num_tgt1] = new_output1.transpose(0, 1)
            output2[:, :num_tgt2] = new_output2.transpose(0, 1)
            
            if self.box_embed is not None:
                if tgt1_reference_points.shape[-1] == 4:
                    new_reference_points1 = self.box_embed[lid](output1) + inverse_sigmoid(tgt1_reference_points)
                    new_reference_points1 = new_reference_points1.sigmoid()
                else:
                    assert tgt1_reference_points.shape[-1] == 2
                    new_reference_points1 = self.box_embed[lid](output1)
                    new_reference_points1[..., :2] = self.box_embed[lid](output1)[..., :2] + inverse_sigmoid(tgt1_reference_points)
                    new_reference_points1 = new_reference_points1.sigmoid()
                tgt1_reference_points = new_reference_points1.detach()

                if tgt2_reference_points.shape[-1] == 4:
                    new_reference_points2 = self.box_embed[lid](output2) + inverse_sigmoid(tgt2_reference_points)
                    new_reference_points2 = new_reference_points2.sigmoid()
                else:
                    assert tgt2_reference_points.shape[-1] == 2
                    new_reference_points2 = self.box_embed[lid](output2)
                    new_reference_points2[..., :2] = self.box_embed[lid](output2)[..., :2] + inverse_sigmoid(tgt2_reference_points)
                    new_reference_points2 = new_reference_points2.sigmoid()
                tgt2_reference_points = new_reference_points2.detach()

            if self.return_intermediate:
                intermediate1.append(output1)
                intermediate2.append(output2)

                intermediate1_reference_points.append(tgt1_reference_points)
                intermediate2_reference_points.append(tgt2_reference_points)

        if self.norm is not None:
            output1 = self.norm(output1)
            output2 = self.norm(output2)
            if self.return_intermediate:
                intermediate1[-1] = output1
                intermediate2[-1] = output2

        if self.return_intermediate:
            return torch.stack(intermediate1), torch.stack(intermediate2), torch.stack(intermediate1_reference_points), torch.stack(intermediate2_reference_points)

        return output1, output2, tgt1_reference_points, tgt2_reference_points

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        num_ps_new_born=args.num_ps_new_born,
        num_cross_attn_layers=args.cross_attn_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        decoder_self_cross=not args.decoder_cross_self,
        sigmoid_attn=args.sigmoid_attn,
        extra_track_attn=args.extra_track_attn,
        memory_bank=args.memory_bank_type == 'MemoryBankFeat'
    )
