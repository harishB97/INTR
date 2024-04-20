# ------------------------------------------------------------------------
# INTR
# Copyright (c) 2023 Imageomics Paul. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
INTR model and loss.
"""
import torch
from torch import nn
import torch.nn.functional as F

import random
from .backbone import build_backbone
from .transformer import build_transformer
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       )

from omegaconf import OmegaConf
from util.node import Node
from util.phylo_utils import construct_phylo_tree, construct_discretized_phylo_tree, set_anclabels_discretized_phylo_tree


class HierINTR(nn.Module):
    """ This is the INTR module that performs explainable image classification """
    def __init__(self, args, backbone, transformer, num_queries, spclabel_to_anclabels):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py (no pos_embed in decoder)
            num_queries: number of classes in the dataset
        """
        super().__init__()
        self.args = args
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        # INTR classification head presence vector
        self.presence_vector = nn.Linear(hidden_dim, 1)

        self.level_to_numclasses = self.get_level_to_numclasses(spclabel_to_anclabels)
        self.numlevels = len(self.level_to_numclasses)

        # TODO: Modify hidden_dim according to num of levels

        for level, numclasses in self.level_to_numclasses.items():
            setattr(self, 'query_embed_'+'level'+str(level)) = nn.Embedding(numclasses, int(hidden_dim // self.numlevels))
        self.query_embed_spc = nn.Embedding(num_queries, int(hidden_dim // self.numlevels))

        queries = []
        for spclabel in spclabel_to_anclabels:
            query = torch.empty(0)
            anclabels = spclabel_to_anclabels[spclabel]
            for level, numclasses in self.level_to_numclasses.items():
                anc_label = anclabels[level]
                anc_query = getattr(self, 'query_embed_'+'level'+str(level))[anc_label]
                query = torch.cat([query, anc_query])
            query = torch.cat([query, self.query_embed_spc[spclabel]])
            queries.append(query)

        self.query_embed = torch.stack(queries)
        # self.query_embed = nn.Embedding(num_queries, int(hidden_dim // self.numlevels))
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def get_level_to_numclasses(self, spclabel_to_anclabels):
        level_to_numclasses = {}
        numlevels = list(spclabel_to_anclabels.values())[0]
        for level in range(numlevels):
            ancclass_indices = set()
            for anclabels in spclabel_to_anclabels.values():
                ancclass_indices.add(anclabels[level])
            numclasses = len(ancclass_indices)
            level_to_numclasses[level] = numclasses
        return level_to_numclasses

    def forward(self, samples: NestedTensor):

        """  The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]

            It returns the following elements:
               - "out": it is a dictnary which currently contains all logit values for for all queries.
                                Shape= [batch_size x num_queries x 1]
               - "encoder_output": it is the output of the transformer encoder which is basically feature map. 
                                Shape= [batch_size x num_features x height x weight]
               - "hs": it is the output of the transformer decoder. These are learned class specific queries. 
                                Shape= [dec_layers x batch_size x num_queries x num_features]
               - "attention_scores": it is attention weight corresponding to each pixel in the encoder  for all heads. 
                                Shape= [dec_layers x batch_size x num_heads x num_queries x height*weight]
               - "avg_attention_scores": it is attention weight corresponding to each pixel in the encoder for avg of all heads. 
                                Shape= [dec_layers x batch_size x num_queries x height*weight]

        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples) # Resnet -> [B, C, H1, W1]

        src, mask = features[-1].decompose()

        assert mask is not None
        hs, encoder_output, attention_scores, avg_attention_scores = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])

        query_logits = self.presence_vector(hs[-1])
        out = {'query_logits': query_logits.squeeze(dim=-1)}

        return out, encoder_output, hs, attention_scores, avg_attention_scores


class SetCriterion(nn.Module):
    """ This class computes the loss for INTR.
        INTR uses only one type of loss i.e., cross entropy loss.
    """
    def __init__(self, args,  model): # weight_dict, losses,
        """ Create the criterion.
        """
        super().__init__()
        self.args = args
        self.model = model

    def get_loss(self, outputs, targets, model):
        """ CE Classification loss
        targets dicts must contain the key "image_label".
        """
        assert 'query_logits' in outputs
        query_logits = outputs['query_logits']
        device = query_logits.device

        target_classes = torch.cat([t['image_label'] for t in targets]) 
        
        criterion = torch.nn.CrossEntropyLoss()
        classification_loss=criterion(query_logits, target_classes)

        losses = {'CE_loss': classification_loss}
        return losses

    def forward(self, outputs, targets, model):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format.
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied. Here we have used only CE loss.
        """
        losses = {}
        losses.update(self.get_loss(outputs, targets, model))
        return losses


def build(args, label_to_spcname):
    """
    In INTR, each query is responsible for learning class specific information.
    So, the `num_queries` here is actually the number of classes in the dataset.
    """

    if args.dataset_name == 'cub':
        args.num_queries = 200
    elif 'cub190' in args.dataset_name:
        args.num_queries = 190
    # elif args.dataset_name== 'bird525':
    #     args.num_queries=525
    # elif args.dataset_name== 'fish':
    #     args.num_queries=183
    # elif args.dataset_name== 'dog':
    #     args.num_queries=120
    # elif args.dataset_name== 'butterfly':
    #     args.num_queries=65
    # elif args.dataset_name== 'pet':
    #     args.num_queries=37
    # elif args.dataset_name== 'car':
    #     args.num_queries=196
    # elif args.dataset_name== 'craft':
    #     args.num_queries=100
    # else:
    #     print ("Enter a valid dataset") 
    #     exit()

    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    phylo_config = OmegaConf.load(args.phylo_config)
    # construct the phylo tree
    assert phylo_config.phyloDistances_string != 'None' # use discretized tree
    root = construct_discretized_phylo_tree(phylo_config.phylogeny_path, phylo_config.phyloDistances_string)
    root.assign_all_descendents()

    # Set ancestor label for each node at each level based on level order traversal
    set_anclabels_discretized_phylo_tree(root)

    spcname_to_leafnode = {node.name: node for node in root.leaf_descendents}
    spcname_to_anclabels = get_spcname_to_anclabels(spcname_to_leafnode)
    spclabel_to_anclabels = {spclabel: spcname_to_anclabels[spcname] for spclabel, spcname in label_to_spcname.items()}

    model = HierINTR(
        args,
        backbone,
        transformer,
        num_queries=args.num_queries,
        spclabel_to_anclabels=spclabel_to_anclabels,
        )

    criterion = SetCriterion(args, model=model)
    criterion.to(device)

    return model, criterion
