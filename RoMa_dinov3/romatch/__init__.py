"""
DINOv3 기반 RoMa 모델을 위한 간단한 인터페이스
"""

import torch
from romatch.models.model_zoo.roma_models import roma_model
from PIL import Image
import numpy as np


def roma_outdoor(device=None, weights_path=None, dinov3_model_name="facebook/dinov3-vit7b16-pretrain-sat493m"):
    """
    DINOv3 기반 RoMa 모델을 로드합니다.
    
    Args:
        device: 사용할 디바이스 (None이면 자동 선택)
        weights_path: 사전 훈련된 가중치 경로 (None이면 기본 가중치 사용)
        dinov3_model_name: 사용할 DINOv3 모델 이름
    
    Returns:
        로드된 RoMa 모델
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 기본 설정
    resolution = (512, 512)  # DINOv3 패치 16에 맞춘 해상도
    upsample_preds = True
    
    # 가중치 로드 (기본값 사용)
    if weights_path is None:
        # 원본 RoMa 가중치를 사용 (DINOv2 부분은 무시됨)
        weights_path = "../RoMa_original/checkpoints/roma_outdoor.pth"
    
    try:
        weights = torch.load(weights_path, map_location=device)
    except FileNotFoundError:
        print(f"Warning: 가중치 파일을 찾을 수 없습니다: {weights_path}")
        print("가중치 없이 모델을 로드합니다.")
        weights = None
    
    # DINOv3 기반 RoMa 모델 생성
    model = roma_model(
        resolution=resolution,
        upsample_preds=upsample_preds,
        device=device,
        weights=weights,
        dinov3_model_name=dinov3_model_name,
        amp_dtype=torch.float16
    )
    
    return model


def match_images(model, img1, img2, device=None):
    """
    두 이미지 간의 매칭을 수행합니다.
    
    Args:
        model: RoMa 모델
        img1: 첫 번째 이미지 (PIL Image 또는 numpy array)
        img2: 두 번째 이미지 (PIL Image 또는 numpy array)
        device: 사용할 디바이스
    
    Returns:
        warp: 변환 필드
        certainty: 확신도 맵
    """
    if device is None:
        device = next(model.parameters()).device
    
    # 이미지를 PIL Image로 변환
    if isinstance(img1, np.ndarray):
        img1 = Image.fromarray(img1)
    if isinstance(img2, np.ndarray):
        img2 = Image.fromarray(img2)
    
    # 이미지를 512x512로 리사이즈
    img1 = img1.resize((512, 512))
    img2 = img2.resize((512, 512))
    
    # 모델 추론
    with torch.no_grad():
        warp, certainty = model.match(img1, img2, device=device)
    
    return warp, certainty


def sample_matches(model, warp, certainty, num_samples=1000):
    """
    매칭 결과에서 샘플을 추출합니다.
    
    Args:
        model: RoMa 모델
        warp: 변환 필드
        certainty: 확신도 맵
        num_samples: 추출할 샘플 수
    
    Returns:
        matches: 매칭된 점들
        certainty_values: 확신도 값들
    """
    with torch.no_grad():
        matches, certainty_values = model.sample(warp, certainty, num_samples=num_samples)
    
    return matches, certainty_values


def to_pixel_coordinates(model, matches, h1, w1, h2, w2):
    """
    정규화된 좌표를 픽셀 좌표로 변환합니다.
    
    Args:
        model: RoMa 모델
        matches: 정규화된 매칭 좌표
        h1, w1: 첫 번째 이미지의 높이, 너비
        h2, w2: 두 번째 이미지의 높이, 너비
    
    Returns:
        kps_source: 첫 번째 이미지의 픽셀 좌표
        kps_target: 두 번째 이미지의 픽셀 좌표
    """
    with torch.no_grad():
        kps_source, kps_target = model.to_pixel_coordinates(matches, h1, w1, h2, w2)
    
    return kps_source, kps_target


# 사용 예시
if __name__ == "__main__":
    # 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = roma_outdoor(device=device)
    
    print("DINOv3 기반 RoMa 모델이 성공적으로 로드되었습니다!")
    print(f"사용 디바이스: {device}")
    print(f"모델 타입: {type(model)}")