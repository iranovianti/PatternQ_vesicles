import torch
import torch.nn as nn
from torchvision import models

class ClassifierModel(nn.Module):
    def __init__(self, arch='densenet121', num_classes=28, pretrained=True, use_softmax=False):
        super().__init__()
        self.arch = arch.lower()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.use_softmax = use_softmax

        self.base, in_features = self._load_base_model()
        self._modify_first_layer()
        self._replace_classifier(in_features)

    def _load_base_model(self):
        weights = 'IMAGENET1K_V1' if self.pretrained else None
        if self.arch == 'densenet121':
            base = models.densenet121(weights=weights)
            return base, base.classifier.in_features
        elif self.arch == 'densenet169':
            base = models.densenet169(weights=weights)
            return base, base.classifier.in_features
        elif self.arch == 'densenet201':
            base = models.densenet201(weights=weights)
            return base, base.classifier.in_features
        elif self.arch == 'resnet34':
            base = models.resnet34(weights=weights)
            return base, base.fc.in_features
        elif self.arch == 'resnet50':
            base = models.resnet50(weights=weights)
            return base, base.fc.in_features
        elif self.arch == 'efficientnet-b0':
            base = models.efficientnet_b0(weights=weights)
            return base, base.classifier[1].in_features
        elif self.arch == 'efficientnet-b1':
            base = models.efficientnet_b1(weights=weights)
            return base, base.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")

    def _modify_first_layer(self):
        if 'densenet' in self.arch:
            self.base.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif 'resnet' in self.arch:
            self.base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif 'efficientnet' in self.arch:
            self.base.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def _replace_classifier(self, in_features):
        activation = nn.Softmax(dim=1) if self.use_softmax else nn.Sigmoid()
        head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, self.num_classes),
            activation
        )
        if 'densenet' in self.arch:
            self.base.classifier = head
        elif 'resnet' in self.arch:
            self.base.fc = head
        elif 'efficientnet' in self.arch:
            self.base.classifier = head

    def forward(self, x):
        return self.base(x)

class RegressorModel(nn.Module):
    def __init__(self, arch='densenet121', pretrained='imagenet', representation_dim=None,
                 freeze='full', custom_weight_path=None, device='cpu', mode='train'):
        super().__init__()
        self.arch = arch.lower()
        self.pretrained = pretrained  # 'imagenet', 'none', or 'custom'
        self.representation_dim = representation_dim
        self.freeze = freeze
        self.custom_weight_path = custom_weight_path
        self.device = device
        self.mode = mode

        self.base, in_features = self._load_base_model()
        
        self.base = self._replace_classifier(self.base, in_features)
        
        if self.mode == 'train':
            if self.pretrained == 'custom' and self.custom_weight_path:
                state_dict = torch.load(self.custom_weight_path, map_location=self.device)
                if 'densenet' in self.arch or "efficientnet" in self.arch:
                    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('base.classifier')}
                elif 'resnet' in self.arch:
                    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('base.fc')}
                self.load_state_dict(state_dict, strict=False)
            self._apply_freezing()


        elif self.mode == 'eval':
            if self.custom_weight_path:
                self.load_state_dict(torch.load(self.custom_weight_path, map_location=self.device))
            self.base.eval()

    def _load_base_model(self):
        weights = 'IMAGENET1K_V1' if self.pretrained == 'imagenet' else None

        # Model init
        if self.arch == 'densenet121':
            base = models.densenet121(weights=weights)
            in_features = base.classifier.in_features
        elif self.arch == 'densenet169':
            base = models.densenet169(weights=weights)
            in_features = base.classifier.in_features
        elif self.arch == 'densenet201':
            base = models.densenet201(weights=weights)
            in_features = base.classifier.in_features
        elif self.arch == 'resnet34':
            base = models.resnet34(weights=weights)
            in_features = base.fc.in_features
        elif self.arch == 'resnet50':
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features
        elif self.arch == 'efficientnet-b0':
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
        elif self.arch == 'efficientnet-b1':
            base = models.efficientnet_b1(weights=weights)
            in_features = base.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")

        self._modify_first_layer(base)
        return base, in_features

    def _modify_first_layer(self, base):
        if 'densenet' in self.arch:
            base.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif 'resnet' in self.arch:
            base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif 'efficientnet' in self.arch:
            base.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def _replace_classifier(self, base, in_features):
        if self.representation_dim:
            head = nn.Sequential(
                nn.Linear(in_features, self.representation_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(self.representation_dim, 1)
            )
        else:
            head = nn.Linear(in_features, 1)

        if 'densenet' in self.arch:
            base.classifier = head
        elif 'resnet' in self.arch:
            base.fc = head
        elif 'efficientnet' in self.arch:
            base.classifier = head

        return base

    def _apply_freezing(self):
        if self.freeze == 'full':
            if 'densenet' in self.arch or 'efficientnet' in self.arch:
                for param in self.base.features.parameters():
                    param.requires_grad = False
            elif 'resnet' in self.arch:
                for name, param in self.base.named_parameters():
                    if not name.startswith("fc"):
                        param.requires_grad = False
        elif self.freeze == 'half':
            if 'densenet' in self.arch or 'efficientnet' in self.arch:
                layers = list(self.base.features.children())
                midpoint = len(layers) // 2
                for layer in layers[:midpoint]:
                    for param in layer.parameters():
                        param.requires_grad = False
            elif 'resnet' in self.arch:
                layers = list(self.base.children())
                midpoint = len(layers) // 2
                for layer in layers[:midpoint]:
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(self, x):
        return self.base(x)
