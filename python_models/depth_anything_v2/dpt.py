import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

try:
    import timm
except ImportError:
    print("Please install timm: pip install timm")
    raise


class DepthAnythingV2(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False
    ):
        super().__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.pretrained = self._make_encoder(
            encoder,
            features,
            use_pretrained=True,
            use_clstoken=use_clstoken
        )
        
        self.depth_head = DPTHead(
            self.pretrained.num_ch_enc,
            features,
            use_bn=use_bn,
            out_channels=out_channels
        )
    
    def _make_encoder(self, encoder, features, use_pretrained, use_clstoken=False):
        if encoder == 'vits':
            pretrained = _make_vit_b16_backbone(
                model_type='dinov2_vits14',
                hooks=[2, 5, 8, 11],
                vit_features=features,
                use_clstoken=use_clstoken
            )
        elif encoder == 'vitb':
            pretrained = _make_vit_b16_backbone(
                model_type='dinov2_vitb14',
                hooks=[2, 5, 8, 11],
                vit_features=features,
                use_clstoken=use_clstoken
            )
        elif encoder == 'vitl':
            pretrained = _make_vit_b16_backbone(
                model_type='dinov2_vitl14',
                hooks=[4, 11, 17, 23],
                vit_features=features,
                use_clstoken=use_clstoken
            )
        elif encoder == 'vitg':
            pretrained = _make_vit_b16_backbone(
                model_type='dinov2_vitg14',
                hooks=[9, 19, 29, 39],
                vit_features=features,
                use_clstoken=use_clstoken
            )
        else:
            raise ValueError(f'Encoder {encoder} not supported.')
        
        return pretrained
    
    def forward(self, x):
        h, w = x.shape[-2:]
        
        features = self.pretrained(x)
        depth = self.depth_head(features)
        depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=True)
        
        return depth
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        import cv2
        import numpy as np
        
        # 预处理
        image, (h, w) = self.image_transform(raw_image, input_size)
        image = torch.from_numpy(image).unsqueeze(0)
        
        # 推理
        if torch.cuda.is_available():
            image = image.cuda()
        
        depth = self.forward(image)[0, 0]
        
        # 后处理
        depth = depth.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = cv2.resize(depth, (w, h))
        
        return depth
    
    def image_transform(self, img, input_size=518):
        import cv2
        import numpy as np
        
        h, w = img.shape[:2]
        
        # Resize
        scale = input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        pad_h = input_size - new_h
        pad_w = input_size - new_w
        img = cv2.copyMakeBorder(
            img, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=[123.675, 116.28, 103.53]
        )
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        # Ensure normalization arrays are also float32 to prevent type promotion to float64
        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = img.transpose(2, 0, 1)
        
        return img, (h, w)


def _make_vit_b16_backbone(
    model_type='dinov2_vitl14',
    hooks=[4, 11, 17, 23],
    vit_features=256,
    use_clstoken=False
):
    pretrained = _make_pretrained_dinov2(
        model_type, hooks=hooks, use_clstoken=use_clstoken
    )
    
    return pretrained


def _make_pretrained_dinov2(model_type, hooks, use_clstoken=False):
    model = torch.hub.load('facebookresearch/dinov2', model_type)
    
    # ViT models output the same dimension at each layer!
    # These are the embedding dimensions, not per-layer dimensions
    features = [1024, 1024, 1024, 1024]
    
    if model_type == "dinov2_vits14":
        features = [384, 384, 384, 384]  # All layers output 384-dim embeddings
    elif model_type == "dinov2_vitb14":
        features = [768, 768, 768, 768]  # All layers output 768-dim embeddings
    elif model_type == "dinov2_vitl14":
        features = [1024, 1024, 1024, 1024]  # All layers output 1024-dim embeddings
    elif model_type == "dinov2_vitg14":
        features = [1536, 1536, 1536, 1536]  # All layers output 1536-dim embeddings
    
    pretrained = make_backbone_default(model, features, hooks=hooks, use_clstoken=use_clstoken)
    
    return pretrained


def make_backbone_default(model, features, hooks, use_clstoken=False):
    pretrained = nn.Module()
    pretrained.model = model
    pretrained.hooks = hooks
    pretrained.features = features
    pretrained.use_clstoken = use_clstoken
    
    pretrained.num_ch_enc = features
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Get intermediate features
        features = self.model.get_intermediate_layers(
            x, self.hooks, return_class_token=self.use_clstoken
        )
        
        if self.use_clstoken:
            # Split features and class tokens
            class_tokens = [f[0] for f in features]
            features = [f[1] for f in features]
        
        # Reshape features
        patch_h, patch_w = h // 14, w // 14
        features = [
            f.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)
            for f in features
        ]
        
        return features
    
    pretrained.forward = lambda x: forward(pretrained, x)
    
    return pretrained


class DPTHead(nn.Module):
    def __init__(self, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024]):
        super().__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels[i], out_channels[i], kernel_size=1, stride=1, padding=0)
            for i in range(len(in_channels))
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1)
        ])
        
        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features // 2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        # Project features
        features = [self.projects[i](f) for i, f in enumerate(features)]
        
        # Resize to common resolution
        features = [self.resize_layers[i](f) for i, f in enumerate(features)]
        
        # Refine features
        layer_1 = self.scratch.layer1_rn(features[0])
        layer_2 = self.scratch.layer2_rn(features[1])
        layer_3 = self.scratch.layer3_rn(features[2])
        layer_4 = self.scratch.layer4_rn(features[3])
        
        path_4 = self.scratch.refinenet4(layer_4)
        path_3 = self.scratch.refinenet3(path_4, layer_3)
        path_2 = self.scratch.refinenet2(path_3, layer_2)
        path_1 = self.scratch.refinenet1(path_2, layer_1)
        
        out = self.scratch.output_conv(path_1)
        
        return out


def _make_scratch(in_channels, out_channels, groups=1, expand=False):
    scratch = nn.Module()
    
    scratch.layer1_rn = nn.Conv2d(in_channels[0], out_channels, kernel_size=3, stride=1, padding=1, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_channels[1], out_channels, kernel_size=3, stride=1, padding=1, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_channels[2], out_channels, kernel_size=3, stride=1, padding=1, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_channels[3], out_channels, kernel_size=3, stride=1, padding=1, groups=groups)
    
    return scratch


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock(features, nn.ReLU(False), deconv=False, bn=use_bn, expand=False, align_corners=True)


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        super().__init__()
        
        self.align_corners = align_corners
        
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
    
    def forward(self, *xs):
        output = xs[0]
        
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = output + res
        
        output = self.resConfUnit2(output)
        output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=self.align_corners)
        output = self.out_conv(output)
        
        return output


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        
        self.bn = bn
        
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not bn)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not bn)
        
        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        
        self.activation = activation
    
    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        
        return out + x

