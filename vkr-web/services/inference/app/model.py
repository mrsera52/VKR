from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F_t


class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F_t.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F_t.relu(out, inplace=True)


class DrawingNet(nn.Module):
    IMG_FEAT_DIM = 256 + 384
    BBOX_FEAT_DIM = 64
    PROJ_DIM = 256

    def __init__(self, num_classes_dict, target_safe_names, nontarget_safe_names,
                 bbox_in_dim=12, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.stage2 = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2))
        self.stage4 = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.MaxPool2d(2))
        self.stage5 = nn.Sequential(
            ResidualBlock(256, 384))
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.bbox_mlp = nn.Sequential(
            nn.Linear(bbox_in_dim, 64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(64, self.BBOX_FEAT_DIM), nn.ReLU(inplace=True))

        self.target_proj = nn.Sequential(
            nn.Linear(self.IMG_FEAT_DIM + self.BBOX_FEAT_DIM, self.PROJ_DIM),
            nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.nontarget_proj = nn.Sequential(
            nn.Linear(self.IMG_FEAT_DIM + self.BBOX_FEAT_DIM, self.PROJ_DIM),
            nn.ReLU(inplace=True), nn.Dropout(dropout))

        self.target_safe = set(target_safe_names)
        self.nontarget_safe = set(nontarget_safe_names)
        self.heads = nn.ModuleDict({
            col: nn.Linear(self.PROJ_DIM, n) for col, n in num_classes_dict.items()
        })

    def forward(self, img, bbox):
        x = self.stem(img)
        x = self.stage2(x)
        x = self.stage3(x)
        f4 = self.stage4(x)
        f5 = self.stage5(f4)
        img_feat = torch.cat([self.pool(f4).flatten(1), self.pool(f5).flatten(1)], dim=1)
        bbox_feat = self.bbox_mlp(bbox)
        feat = torch.cat([img_feat, bbox_feat], dim=1)
        t_feat = self.target_proj(feat)
        n_feat = self.nontarget_proj(feat)
        return {col: head(t_feat if col in self.target_safe else n_feat)
                for col, head in self.heads.items()}
