import math
import copy

import torch
from torch.nn.functional import relu, interpolate
from torch import nn


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeformableDETR(nn.Module):

    def __init__(self,
                 backbone,
                 position_encoding,
                 transformer,
                 num_classes=9,
                 num_queries=300,
                 num_feature_levels=4):
        super().__init__()
        # Network hyperparameters
        self.hidden_dim = transformer.hidden_dim
        self.num_feature_levels = num_feature_levels
        self.num_queries = num_queries
        self.num_classes = num_classes
        # Backbone: multiscale outputs backbone network
        self.backbone = backbone
        # Build input projections
        self.input_proj = self._build_input_projections()
        # Position encoding
        self.position_encoding = position_encoding
        # Deformable transformer
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim * 2)
        self.transformer = transformer
        # Prediction of class and box
        self.class_embed = nn.Linear(self.hidden_dim, self.num_classes)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        # Initialize parameters
        self._init_params()
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(transformer.decoder.num_layers)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(transformer.decoder.num_layers)])

    def _build_input_projections(self):
        input_proj_list = []
        if self.num_feature_levels > 1:
            for i in range(self.backbone.num_outputs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(self.backbone.num_channels[i], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            in_channels = self.backbone.num_channels[-1]
            for _ in range(self.num_feature_levels - self.backbone.num_outputs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
        else:
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(self.backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                nn.GroupNorm(32, self.hidden_dim),
            ))
        return nn.ModuleList(input_proj_list)

    def _init_params(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)

    @staticmethod
    def inverse_sigmoid(x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)

    def forward(self, images, masks):
        # Backbone forward
        features = self.backbone(images)
        # Prepare input features for transformer
        src_list, mask_list = [], []
        for i, feature in enumerate(features):
            src = self.input_proj[i](feature)
            mask = interpolate(masks[None].float(), size=feature.shape[-2:]).to(torch.bool)[0]
            src_list.append(src)
            mask_list.append(mask)
        if self.num_feature_levels > len(features):
            for i in range(len(features), self.num_feature_levels):
                src = self.input_proj[i](features[-1]) if i == len(features) else self.input_proj[i](src_list[-1])
                mask = interpolate(masks[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                src_list.append(src)
                mask_list.append(mask)
        pos_list = [self.position_encoding(src, mask) for src, mask in zip(src_list, mask_list)]
        query_embeds = self.query_embed.weight
        # Transformer forward
        hs, init_reference, inter_references, _, _, inter_memory, inter_object_query = self.transformer(
            src_list,
            mask_list,
            pos_list,
            query_embeds
        )
        # Prepare outputs
        outputs_classes, outputs_coords = [], []
        for lvl in range(hs.shape[0]):
            outputs_class = self.class_embed[lvl](hs[lvl])
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = self.inverse_sigmoid(reference)
            tmp = self.bbox_embed[lvl](hs[lvl])
            tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        out = {
            'logit_all': outputs_class,
            'boxes_all': outputs_coord,
            'features': features[-1].detach()
        }
        return out
