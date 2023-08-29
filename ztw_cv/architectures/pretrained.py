from functools import partial

import torch
import torchvision
from torchvision.models import ViT_B_16_Weights, EfficientNet_V2_S_Weights, EfficientNet_B0_Weights, \
    ConvNeXt_Tiny_Weights, Swin_V2_S_Weights


def get_efficientnet_b0():
    model = torchvision.models.efficientnet_b0(EfficientNet_B0_Weights.IMAGENET1K_V1, progress=False)

    def forward_generator(self, x):
        for stage in self.features:
            if isinstance(stage, torch.nn.Sequential):
                for block in stage:
                    x = block(x)
                    if isinstance(block, (torchvision.models.efficientnet.FusedMBConv,
                                          torchvision.models.efficientnet.MBConv)):
                        x = yield x, None
            else:
                x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    model.input_size = 224
    model.input_channels = 3
    model.number_of_classes = 1000
    return model


def get_efficientnet_v2_s():
    model = torchvision.models.efficientnet_v2_s(EfficientNet_V2_S_Weights.IMAGENET1K_V1, progress=False)

    def forward_generator(self, x):
        for stage in self.features:
            if isinstance(stage, torch.nn.Sequential):
                for block in stage:
                    x = block(x)
                    if isinstance(block, (torchvision.models.efficientnet.FusedMBConv,
                                          torchvision.models.efficientnet.MBConv)):
                        x = yield x, None
            else:
                x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    model.input_size = 384
    model.input_channels = 3
    model.number_of_classes = 1000
    return model


def get_convnext_t():
    model = torchvision.models.convnext_tiny(ConvNeXt_Tiny_Weights.IMAGENET1K_V1, progress=False)

    def forward_generator(self, x):
        for stage in self.features:
            if isinstance(stage, torch.nn.Sequential):
                for block in stage:
                    x = block(x)
                    x = yield x, None
            else:
                x = stage(x)
                # x = yield x, None
        x = self.avgpool(x)
        x = self.classifier(x)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    model.input_size = 224
    model.input_channels = 3
    model.number_of_classes = 1000
    return model


def get_vit_b_16():
    model = torchvision.models.vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1, progress=False)

    def forward_generator(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        # BEGIN ENCODER
        # equivalent to: x = self.encoder(x)
        x = x + self.encoder.pos_embedding
        x = self.encoder.dropout(x)
        # go through encoder blocks
        for block in self.encoder.layers:
            x = block(x)
            x = yield x, None
        x = self.encoder.ln(x)
        # END OF ENCODER
        # classifier token
        x = x[:, 0]
        x = self.heads(x)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    model.input_size = 224
    model.input_channels = 3
    model.number_of_classes = 1000
    return model


def get_swin_v2_s():
    model = torchvision.models.swin_v2_s(Swin_V2_S_Weights.IMAGENET1K_V1, progress=False)

    def forward_generator(self, x):
        for stage in self.features:
            if isinstance(stage, torch.nn.Sequential):
                for block in stage:
                    x = block(x)
                    x = yield x, None
            else:
                x = stage(x)
                # x = yield x, None
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    model.input_size = 256
    model.input_channels = 3
    model.number_of_classes = 1000
    return model
