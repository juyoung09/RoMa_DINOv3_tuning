import warnings
import torch.nn as nn
import torch
from romatch.models.matcher import *
from romatch.models.transformer import Block, TransformerDecoder, MemEffAttention
from romatch.models.encoders import *
from romatch.models.encoders import CNNandDinov3
from romatch.models.tiny import TinyRoMa

def tiny_roma_v1_model(weights = None, freeze_xfeat=False, exact_softmax=False, xfeat = None):
    model = TinyRoMa(
        xfeat = xfeat,
        freeze_xfeat=freeze_xfeat, 
        exact_softmax=exact_softmax)
    if weights is not None:
        model.load_state_dict(weights)
    return model

def load_roma_weights_except_coarse(model, original_weights_path):
    """
    원본 RoMa 가중치를 로드하되, coarse level (DINOv3 부분)은 제외하고 나머지 부분만 로드합니다.
    """
    print("🔄 원본 RoMa 가중치 로드 중 (DINOv3 coarse level 제외)...")
    
    # 원본 가중치 로드
    original_weights = torch.load(original_weights_path, map_location='cpu')
    
    # 현재 모델의 state_dict 가져오기
    current_state_dict = model.state_dict()
    
    # 로드할 가중치 필터링 (DINOv3 관련 부분 제외)
    filtered_weights = {}
    skipped_keys = []
    
    for key, value in original_weights.items():
        # DINOv3 관련 키들 제외 (체크포인트 selective 로딩)
        if any(skip_pattern in key for skip_pattern in [
            'encoder.cnn.',  # VGG19 부분은 제외 (DINOv3로 대체됨)
            'encoder.dinov2_model',  # DINOv2 모델 부분 (기존)
            'encoder.dinov3_model',  # DINOv3 모델 부분 (새로 추가)
            'encoder._dinov3_model',  # DINOv3 모델 부분 (새로 추가)
            # 'decoder.proj.16.0.weight',  # proj16은 동적 설정으로 로드 가능
        ]):
            skipped_keys.append(key)
            continue
            
        # 현재 모델에 해당 키가 있고 크기가 맞는 경우만 로드
        if key in current_state_dict:
            if current_state_dict[key].shape == value.shape:
                filtered_weights[key] = value
            else:
                # proj16 키는 강제 로드/변환 (동적 차원으로 인한 크기 불일치 허용)
                if 'proj.16' in key and value.shape[1] != current_state_dict[key].shape[1]:
                    print(f"  ⚠️  proj16 크기 불일치로 건너뜀: {key} (원본: {value.shape}, 현재: {current_state_dict[key].shape})")
                    skipped_keys.append(key)
                    continue
                else:
                    print(f"  ⚠️  크기 불일치로 건너뜀: {key} (원본: {value.shape}, 현재: {current_state_dict[key].shape})")
                    skipped_keys.append(key)
        else:
            print(f"  ⚠️  키가 없어서 건너뜀: {key}")
            skipped_keys.append(key)
    
    # 필터링된 가중치 로드
    model.load_state_dict(filtered_weights, strict=False)
    
    print(f"✅ 가중치 로드 완료!")
    print(f"  - 로드된 키: {len(filtered_weights)}개")
    print(f"  - 건너뛴 키: {len(skipped_keys)}개")
    
    if skipped_keys:
        print(f"  - 건너뛴 키들: {skipped_keys[:5]}{'...' if len(skipped_keys) > 5 else ''}")
    
    return model, skipped_keys

def roma_model(resolution, upsample_preds, device = None, weights=None, dinov3_model_name="facebook/dinov3-vit7b16-pretrain-sat493m", amp_dtype: torch.dtype=torch.float16, original_roma_weights=None, **kwargs):
    # 입력 해상도 512x512로 변경 (패치 16 배수)
    resolution = (512, 512)
    h, w = resolution
    
    # romatch weights and dinov3 weights are loaded seperately, as dinov3 weights are not parameters
    #torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul TODO: these probably ruin stuff, should be careful
    #torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    gp_dim = 512
    feat_dim = 512
    decoder_dim = gp_dim + feat_dim
    cls_to_coord_res = 64
    coordinate_decoder = TransformerDecoder(
        nn.Sequential(*[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]), 
        decoder_dim, 
        cls_to_coord_res**2 + 1,
        is_classifier=True,
        amp = True,
        pos_enc = False,)
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True
    
    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512+128+(2*7+1)**2,
                2 * 512+128+(2*7+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=128,
                local_corr_radius = 7,
                corr_in_other = True,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "8": ConvRefiner(
                2 * 512+64+(2*3+1)**2,
                2 * 512+64+(2*3+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=64,
                local_corr_radius = 3,
                corr_in_other = True,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "4": ConvRefiner(
                2 * 256+32+(2*2+1)**2,
                2 * 256+32+(2*2+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=32,
                local_corr_radius = 2,
                corr_in_other = True,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "2": ConvRefiner(
                2 * 64+16,
                128+16,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=16,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "1": ConvRefiner(
                2 * 9 + 6,
                24,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks = hidden_blocks,
                displacement_emb = displacement_emb,
                displacement_emb_dim = 6,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
        }
    )
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"16": gp16})
    
    # 동적 proj16 (DINOv3 차원 4096 적응)
    from transformers import AutoModel
    dinov3_model = AutoModel.from_pretrained(dinov3_model_name)
    hidden_size = dinov3_model.config.hidden_size  # 4096
    proj16 = nn.Sequential(nn.Conv2d(hidden_size, 512, 1, 1), nn.BatchNorm2d(512))
    proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
    proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
    proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
    proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))
    proj = nn.ModuleDict({
        "16": proj16,
        "8": proj8,
        "4": proj4,
        "2": proj2,
        "1": proj1,
        })
    displacement_dropout_p = 0.0
    gm_warp_dropout_p = 0.0
    decoder = Decoder(coordinate_decoder, 
                      gps, 
                      proj, 
                      conv_refiner, 
                      detach=True, 
                      scales=["16", "8", "4", "2", "1"], 
                      displacement_dropout_p = displacement_dropout_p,
                      gm_warp_dropout_p = gm_warp_dropout_p)
    
    encoder = CNNandDinov3(
        cnn_kwargs = dict(
            pretrained=False,
            amp = True),
        amp = True,
        use_vgg = True,
        dinov3_model_name = dinov3_model_name,
        amp_dtype=amp_dtype,
    )
    h,w = resolution
    symmetric = True
    attenuate_cert = True
    sample_mode = "threshold_balanced"
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w, upsample_preds=upsample_preds, 
                                symmetric = symmetric, attenuate_cert = attenuate_cert, sample_mode = sample_mode, **kwargs).to(device)
    
    # 원본 RoMa 가중치 로드 (DINOv3 coarse level 제외)
    if original_roma_weights is not None:
        matcher, skipped_keys = load_roma_weights_except_coarse(matcher, original_roma_weights)
        
        # proj16.0.weight가 스킵되었을 때만 Kaiming 초기화 적용
        if 'decoder.proj.16.0.weight' in skipped_keys:
            print("🔄 proj16.0.weight가 스킵되어 Kaiming 초기화 적용")
            nn.init.kaiming_normal_(matcher.decoder.proj['16'][0].weight, mode='fan_out', nonlinearity='relu')
            if matcher.decoder.proj['16'][0].bias is not None:
                nn.init.constant_(matcher.decoder.proj['16'][0].bias, 0)
    else:
        # 가중치 로드가 없을 때는 항상 Kaiming 초기화 적용
        nn.init.kaiming_normal_(matcher.decoder.proj['16'][0].weight, mode='fan_out', nonlinearity='relu')
        if matcher.decoder.proj['16'][0].bias is not None:
            nn.init.constant_(matcher.decoder.proj['16'][0].bias, 0)
    
    # 추가 가중치 로드 (있는 경우)
    if weights is not None:
        matcher.load_state_dict(weights)
    
    return matcher