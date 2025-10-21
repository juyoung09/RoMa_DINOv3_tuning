from typing import Optional, Union
import torch
from torch import device
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import gc
from romatch.utils.utils import get_autocast_params


class ResNet50(nn.Module):
    def __init__(self, pretrained=False, high_res = False, weights = None, 
                 dilation = None, freeze_bn = True, anti_aliased = False, early_exit = False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        if dilation is None:
            dilation = [False,False,False]
        if anti_aliased:
            pass
        else:
            if weights is not None:
                self.net = tvm.resnet50(weights = weights,replace_stride_with_dilation=dilation)
            else:
                self.net = tvm.resnet50(pretrained=pretrained,replace_stride_with_dilation=dilation)
            
        self.high_res = high_res
        self.freeze_bn = freeze_bn
        self.early_exit = early_exit
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            net = self.net
            feats = {1:x}
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            feats[2] = x 
            x = net.maxpool(x)
            x = net.layer1(x)
            feats[4] = x 
            x = net.layer2(x)
            feats[8] = x
            if self.early_exit:
                return feats
            x = net.layer3(x)
            feats[16] = x
            x = net.layer4(x)
            feats[32] = x
            return feats

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                pass

class VGG19(nn.Module):
    def __init__(self, pretrained=False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        # AMP 비활성화하여 타입 불일치 방지
        feats = {}
        scale = 1
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                feats[scale] = x
                scale = scale*2
            x = layer(x)
        return feats

def get_autocast_params(device, amp, amp_dtype):
    """AMP 설정을 위한 파라미터 반환"""
    if device.type == 'cuda' and amp:
        return 'cuda', True, amp_dtype
    elif device.type == 'cpu' and amp:
        return 'cpu', True, amp_dtype
    else:
        return device.type, False, torch.float32

class CNNandDinov3(nn.Module):
    def __init__(self, cnn_kwargs=None, amp=False, use_vgg=False, dinov3_model_name="facebook/dinov3-vit7b16-pretrain-sat493m", amp_dtype=torch.float16):
        super().__init__()
        
        # DINOv3 로드 및 frozen (RoMa 스타일)
        from transformers import AutoModel, AutoImageProcessor
        
        self.dinov3_model = AutoModel.from_pretrained(dinov3_model_name)
        self.dinov3_processor = AutoImageProcessor.from_pretrained(dinov3_model_name)
        
        # DINOv3 완전히 frozen (RoMa 논문과 동일)
        self.dinov3_model.eval()
        for param in self.dinov3_model.parameters():
            param.requires_grad = False  # DINOv3 frozen (RoMa 스타일)
        
        # 나머지 CNN 부분 유지 (미세 특징용)
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        if not use_vgg:
            self.cnn = ResNet50(**cnn_kwargs)
        else:
            self.cnn = VGG19(**cnn_kwargs)
        
        self.amp = amp
        self.amp_dtype = amp_dtype
        
        # GPU 이동
        if torch.cuda.is_available():
            self.cnn = self.cnn.cuda()
        
        # DDP hack
        self._dinov3_model = [self.dinov3_model]

    def train(self, mode: bool = True):
        # CNN 부분만 train/eval 모드 전환
        return self.cnn.train(mode)
    
    def forward(self, x, upsample=False):
        B, C, H, W = x.shape
        
        # --- 경로 1: CNN 피라미드 추출 (미세 특징) ---
        feature_pyramid = self.cnn(x)
        
        if not upsample:
            with torch.no_grad():
                # DINOv3를 현재 입력과 동일한 디바이스로 이동
                if self._dinov3_model[0].device != x.device:
                    self._dinov3_model[0] = self._dinov3_model[0].to(x.device)

                # --- 경로 2: DINOv3 coarse 특징 추출 (frozen) ---
                # 입력 처리: [0,1] 범위로 변환 후 프로세서 적용
                x_denorm = (x + 1) / 2  # RoMa가 [-1,1] 정규화라면 [0,1]로
                images = [x_denorm[i].permute(1,2,0).cpu().numpy() for i in range(B)]
                inputs = self.dinov3_processor(images, return_tensors="pt").to(x.device)
                
                # DINOv3 출력 (frozen)
                dinov3_output = self._dinov3_model[0](**inputs)
                last_hidden_state = dinov3_output.last_hidden_state
                
                # CLS(1) + 레지스터(4) 제외, 패치 토큰만 추출
                patch_tokens = last_hidden_state[:, 5:]  # [B, num_patches, 4096]
                
                # 패치 크기 확인 및 reshaping
                patch_size = self.dinov3_model.config.patch_size  # 16
                target_patches = (H // patch_size) * (W // patch_size)  # 정확한 계산
                B, num_patches, hidden_dim = patch_tokens.shape
                
                # 패치 수 불일치 시 패딩/슬라이싱
                if num_patches > target_patches:
                    patch_tokens = patch_tokens[:, :target_patches, :]
                elif num_patches < target_patches:
                    padding = torch.zeros(B, target_patches - num_patches, hidden_dim, 
                                          device=patch_tokens.device, dtype=patch_tokens.dtype)
                    patch_tokens = torch.cat([patch_tokens, padding], dim=1)
                
                # 특징 맵으로 변환: 32x32 (512x512 입력, 패치 16)
                features_32x32 = patch_tokens.permute(0,2,1).reshape(B, hidden_dim, H//patch_size, W//patch_size)
                
                # 32x32 -> 35x35로 보간하여 원본 맵 크기 맞춤 (512x512 입력 시)
                features_35x35 = F.interpolate(features_32x32, size=(35, 35), mode='bilinear', align_corners=False)
                
                # 기존 VGG19 특징을 DINOv3 특징으로 교체
                feature_pyramid[16] = features_35x35.float()
                
                del dinov3_output
                
        return feature_pyramid


# CNNandDinov3Real은 CNNandDinov3의 별칭입니다
CNNandDinov3Real = CNNandDinov3