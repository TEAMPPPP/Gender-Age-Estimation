from torch import nn
import torch
import torch.nn.functional as F
import timm

algc = False

class MSGFAM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(MSGFAM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )

        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        
        self.scale_process = nn.Sequential(
                                    BatchNorm(branch_planes*4, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes*4, branch_planes*4, kernel_size=3, padding=1, groups=4, bias=False),
                                    )

      
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(F.interpolate(self.scale1(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale2(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale3(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        
        scale_out = self.scale_process(torch.cat(scale_list, 1))
       
        out = self.compression(torch.cat([x_,scale_out], 1)) + self.shortcut(x)
        return out
    
class GAENet(nn.Module):
    def __init__(self):
        super(GAENet, self).__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0, features_only=True)
        # self.backbone = timm.create_model('convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384', pretrained=True, num_classes=0, features_only=True)

        input_channels = 1024 

        self.intermediate_convs = nn.Sequential(
            nn.Conv2d(input_channels, 1024, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        # MSGFAM 모듈
        self.msgfam = MSGFAM(inplanes=256, branch_planes=64, outplanes=256, BatchNorm=nn.BatchNorm2d)

        # 최종 예측을 위한 MLP
        self.mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.age_branch = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.gender_branch = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.backbone(x)[-1]
        features = features.permute(0, 3, 1, 2)  # Reshapes the tensor from [8, 7, 7, 1024] to [8, 1024, 7, 7]

        features_reduced = self.intermediate_convs(features)
        msgfam_features = self.msgfam(features_reduced)
        pooled_features = torch.flatten(nn.functional.adaptive_avg_pool2d(msgfam_features, (1, 1)), 1)

        mlp_features = self.mlp(pooled_features)

        age_output = self.age_branch(mlp_features)
        gender_output = self.gender_branch(mlp_features)

        return age_output, gender_output