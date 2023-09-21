import copy

import torch
import torch.nn as nn
import torchvision


def mlp(in_c, hid_c, out_c, n_layer=0):
    layers = [nn.Linear(in_c, hid_c),
              nn.BatchNorm1d(hid_c),
              nn.ReLU(True)]
    for _ in range(n_layer):
        layers += [
            nn.Linear(hid_c, hid_c),
            nn.BatchNorm1d(hid_c),
            nn.ReLU(True)
        ]
    layers.append(nn.Linear(hid_c, out_c))
    return nn.Sequential(*layers)


class AttributeClassifier(nn.Module):
    def __init__(self, backbone='r34'):
        super(AttributeClassifier, self).__init__()
        import torchvision
        if backbone == 'r34':
            backbone = torchvision.models.resnet34(pretrained=True)
            in_dim = 512
        elif backbone == 'r50':
            backbone = torchvision.models.resnet50(pretrained=True)
            in_dim = 2048
        elif backbone == 'r18':
            backbone = torchvision.models.resnet18(pretrained=True)
        else:
            raise NotImplementedError
        self.extractor = copy.deepcopy(backbone)
        self.extractor.fc = nn.Identity()

        self.attr_heads = nn.ModuleList([mlp(in_dim, in_dim, 1, 2) for _ in range(40)])

        self.age_heads = copy.deepcopy(backbone)
        self.age_heads.fc = nn.Linear(in_dim, 6)

    def forward(self, x):
        outs = []

        features = self.extractor(x)
        for i, head in enumerate(self.attr_heads):
            outs.append(head(features))

        outs.append(self.age_heads(x))
        outs = torch.cat(outs, dim=1)

        preds = (torch.sigmoid(outs) > 0.5).float()
        preds = torch.cat([preds[:, :40], preds[:, 40:].sum(dim=1).unsqueeze(1)], dim=1)
        return outs, preds

    def forward_attr(self, x):
        features = self.extractor(x)
        outs = []
        for i, head in enumerate(self.attr_heads):
            outs.append(head(features))

        outs = torch.cat(outs, dim=1)
        preds = (torch.sigmoid(outs) > 0.5).float()
        return outs, preds

    def forward_age(self, x):
        outs = self.age_heads(x)
        preds = (torch.sigmoid(outs) > 0.5).float().sum(dim=1)
        return outs, preds
    

class AttributeEstimator(nn.Module):
    def __init__(self, backbone='resnet34',attribute_dim=40):
        super(AttributeEstimator, self).__init__()
        self.attribute_dim = attribute_dim
        if backbone == 'alexnet':
            model = torchvision.models.alexnet(pretrained=False)
            model.classifier[-1] = nn.Linear(4096, attribute_dim)
        elif backbone == 'vgg16':
            model = torchvision.models.vgg16(pretrained=False)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, attribute_dim)
        elif backbone == 'vgg16bn':
            model = torchvision.models.vgg16_bn(pretrained=False)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, attribute_dim)
        elif backbone == 'resnet18':
            model = torchvision.models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, attribute_dim)
        elif backbone == 'resnet34':
            model = torchvision.models.resnet34(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, attribute_dim)
        elif backbone == 'resnet50':
            model = torchvision.models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, attribute_dim)
        elif backbone == 'resnet101':
            model = torchvision.models.resnet101(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, attribute_dim)
        elif backbone == 'inceptionv3':
            model = torchvision.models.inception_v3(pretrained=False, aux_logits=False)
            model.fc = nn.Linear(model.fc.in_features, attribute_dim)
        else:
            raise NotImplementedError('backbone [%s] is not implemented' % backbone)
        self.backbone = model
        self.sigmoid = nn.Sigmoid()
    def forward(self,img,latent=None):
        bs = img.size(0)
        out = self.sigmoid(self.backbone(img))
        out = out.view(bs, self.attribute_dim)
        return out