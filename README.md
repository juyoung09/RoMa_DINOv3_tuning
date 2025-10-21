# RoMa DINOv3 Integration

이 프로젝트는 RoMa 모델의 DINOv2 인코더를 DINOv3로 교체한 비교연구입니다.

## 📁 폴더 구조

```
RoMa_DINOv3/
├── RoMa_original/          # 원본 RoMa 모델 (DINOv2 사용)
├── RoMa_dinov3/           # DINOv3 통합 RoMa 모델
├── processed_refine_data/  # 평가 데이터셋
├── preprocessing.py        # 전처리 모듈
├── compare_roma_models.py  # 비교평가 스크립트
├── hf_token.txt           # Hugging Face 토큰
└── README.md              # 이 파일
```

## 🔧 모델 구조

### 원본 RoMa (RoMa_original)
- **인코더**: VGG19 + DINOv2 ViT-L/14
- **디코더**: RoMa Decoder + Refiner
- **특징**: Fine-tuned된 DINOv2 사용

### DINOv3 RoMa (RoMa_dinov3)
- **인코더**: VGG19 + DINOv3 ViT-L/16 Satellite
- **디코더**: RoMa Decoder + Refiner (체크포인트에서 로드)
- **특징**: 사전학습된 DINOv3 Satellite 모델 사용

## 🚀 사용법

### 1. 환경 설정
```bash
# 가상환경 활성화 (roma_env 사용)
source ../RoMa/roma_env/bin/activate

# 필요한 패키지 설치
pip install transformers accelerate torchmetrics termcolor
```

### 2. 모델 비교평가
```bash
# 기본 평가 (100개 샘플)
python3 compare_roma_models.py

# 500개 샘플로 평가
python3 compare_roma_models.py --samples 500

# 결과를 특정 파일에 저장
python3 compare_roma_models.py --output my_results.json
```

## 📊 평가 메트릭

- **RMSE**: Root Mean Square Error
- **PCK@1,3,5px**: Percentage of Correct Keypoints
- **Translation Error**: 평균 이동 오차
- **Rotation Error**: 평균 회전 오차  
- **Scale Error**: 평균 크기 오차
- **Success Rate**: 성공률 (RMSE < 100px)

## 🔍 핵심 차이점

### DINOv2 vs DINOv3
| 특징 | DINOv2 | DINOv3 |
|------|--------|--------|
| Patch Size | 14×14 | 16×16 |
| 이미지 크기 | 518×518 | 560×560 |
| 사전학습 데이터 | ImageNet | Satellite (SAT-493M) |
| 특징 차원 | 1024 | 1024 |
| 공간 해상도 | 40×40 | 35×35 → 40×40 (interpolate) |

### 모델 로딩 방식
- **원본 RoMa**: DINOv2 체크포인트 로드
- **DINOv3 RoMa**: Hugging Face에서 DINOv3 모델 로드

## 📈 예상 결과

DINOv3 Satellite 모델은 위성 영상에 특화되어 있어 다음과 같은 성능 향상을 기대할 수 있습니다:

- **더 정확한 특징 추출**: 위성 영상 도메인 특화
- **향상된 일반화**: 최신 아키텍처 활용
- **안정적인 매칭**: 개선된 패치 처리

## 🛠️ 기술적 세부사항

### DINOv3 통합 과정
1. DINOv3 모델을 Hugging Face에서 로드
2. VGG19 + Decoder/Refiner는 기존 체크포인트에서 로드
3. DINOv3의 35×35 특징을 40×40으로 보간하여 호환성 확보

### 전처리 차이
- **DINOv2**: ImageNet 정규화 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **DINOv3**: Satellite 특화 정규화

## 📝 참고사항

- DINOv3 모델은 `hf_token.txt`의 Hugging Face 토큰을 사용하여 로드됩니다
- 평가는 `processed_refine_data` 폴더의 데이터를 사용합니다
- GPU 사용 시 CUDA 가속이 자동으로 적용됩니다

## 🔗 관련 링크

- [RoMa 원본 프로젝트](https://github.com/Parskatt/RoMa)
- [DINOv3 논문](https://arxiv.org/abs/2404.14219)
- [DINOv3 Hugging Face](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m)
