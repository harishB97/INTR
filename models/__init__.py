# ------------------------------------------------------------------------
# INTR
# Copyright (c) 2023 Imageomics Paul. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
from .intr import build
from .hier_intr import build as build_hierINTR

def build_model(args):
    return build(args)

def build_model_hierINTR(args, label_to_spcname):
    return build_hierINTR(args, label_to_spcname)
