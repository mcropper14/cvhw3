import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]


#ASPP = parallel 3×3 atrous convs with different dilation rates + a 1×1 branch + an image level pooling branch for global context




class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__() #for some reason this gave error 
        
        #3x3 convolution with given dilation rate
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ASPPPooling, self).__init__()
        #forgot this 
        self.gp = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU())
        #raise NotImplementedError
        # TODO Problem 2.1
        # ================================================================================ #

    def forward(self, x):
        h, w = x.shape[-2:]
        y = self.gp(x)
        y = self.conv(y)
        return F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
        # TODO Problem 2.1
        # ================================================================================ #
        raise NotImplementedError


#SPP {1×1, 3×3@rate6, 3×3@rate12, 3×3@rate18, image pooling}, concat, 1×1 fuse, classifier logits, upsample to input size.

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        #branches = [1x1, 3x3@rate1, 3x3@rate2, 3x3@rate3, image pooling]
        rate1, rate2, rate3 = tuple(atrous_rates)

        branches = []
        # 1x1 branch
        branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )
        #3x3 atrous branches
        branches.append(ASPPConv(in_channels, out_channels, dilation=rate1))
        branches.append(ASPPConv(in_channels, out_channels, dilation=rate2))
        branches.append(ASPPConv(in_channels, out_channels, dilation=rate3))
        # image pooling branch
        branches.append(ASPPPooling(in_channels, out_channels))

        self.branches = nn.ModuleList(branches)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(self.branches), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )


        # TODO Problem 2.1
        # ================================================================================ #
        #raise NotImplementedError
        

    def forward(self, x):
        # TODO Problem 2.1
        # ================================================================================ #
        #raise NotImplementedError
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return self.project(x)

# DeepLabV3 model
#uses backbone to extract features, then classifier (ASPP + convs) to predict logits
#SPP {1×1, 3×3@rate6, 3×3@rate12, 3×3@rate18, image pooling}, concat, 1×1 fuse, classifier logits, upsample to input size.
#
class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    #pass
    #def __init__(self, backbone: nn.Module, classifier: nn.Module, aux_classifier: nn.Module | None = None):
    def __init__(self, backbone, classifier, aux_classifier=None):
        #super().__init__(backbone, classifier, aux_classifier=aux_classifier)
        
        super().__init__(backbone, classifier)
       


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):

        #ASPP on high level features, 3x3 conv, logits 

        super(DeepLabHead, self).__init__()
        # TODO Problem 2.2
        # The model should have the following 3 arguments
        #   in_channels: number of input channels
        #   num_classes: number of classes for prediction
        #   aspp_dilate: atrous_rates for ASPP
        #   
        # ================================================================================ #
        self.aspp = ASPP(in_channels, aspp_dilate, out_channels=256)
     
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        #raise NotImplementedError

        
        self._init_weight()

    def forward(self, feature):
        # TODO Problem 2.2
        # ================================================================================ #
        x = self.aspp(feature)
        x = self.classifier(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


#encoder - decoder 
#decoder - upsample encoder features. concentate with low level features from earlier layer, refine with convs, predict logits

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        #ASPP encodr + decoder with low level features
        super(DeepLabHeadV3Plus, self).__init__()
        # TODO Problem 2.2
        # The model should have the following 4 arguments
        #   in_channels: number of input channels
        #   low_level_channels: number of channels for project
        #   num_classes: number of classes for prediction
        #   aspp_dilate: atrous_rates for ASPP
        #   
        # ================================================================================ #
        self.aspp = ASPP(in_channels, aspp_dilate, out_channels=256)

        #project low-level to a small number of channels
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )


        self.refine = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)




        self._init_weight()

    def forward(self, feature):
        # TODO Problem 2.2
        # ================================================================================ #
        #raise NotImplementedError
        high = feature["out"]
        low = feature["low_level"]

        x = self.aspp(high)
        # upsample ASPP output by 4 to roughly match low-level stride
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)

        low = self.project(low)
        x = torch.cat([x, low], dim=1)
        x = self.refine(x)
        x = self.classifier(x)
        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
