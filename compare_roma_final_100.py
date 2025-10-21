#!/usr/bin/env python3
"""
DINOv2 vs DINOv3 RoMa 모델 간단 비교 평가 스크립트 (100개 랜덤 샘플)
"""

import os
import sys
import json
import cv2
import numpy as np
import torch
import random
from PIL import Image
from tqdm import tqdm

# ============================================================================
# 설정 (Configuration)
# ============================================================================
class Config:
    # --- 데이터 및 모델 경로 ---
    DATA_DIR = "./processed_refine_data"  # 전처리된 데이터 폴더
    LABEL_MAPPING_FILE = os.path.join(DATA_DIR, "label_mapping.json")
    
    # --- 평가 설정 ---
    TARGET_SIZE = 512  # 이미지를 리사이즈할 크기
    RANSAC_THRESHOLD = 3.0  # RANSAC 이상치 제거 임계값
    MIN_MATCHES = 8  # Homography 추정을 위한 최소 매칭 수
    NUM_RANDOM_SAMPLES = 100  # 랜덤 샘플 수

    # --- 평가지표 임계값 ---
    PCK_THRESHOLDS = [1, 3, 5]  # PCK 계산을 위한 픽셀 임계값


# ============================================================================
# 유틸리티 함수 (Utility Functions)
# ============================================================================
def load_json_file(filepath):
    """JSON 파일을 로드합니다."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)

def get_random_samples(data_dir, num_samples=100):
    """데이터셋에서 랜덤 샘플을 선정합니다."""
    # 모든 pair 디렉토리 찾기
    pair_dirs = [d for d in os.listdir(data_dir) if d.startswith('pair_')]
    pair_dirs.sort()
    
    print(f"총 {len(pair_dirs)}개의 pair 중에서 {num_samples}개를 랜덤으로 선정합니다.")
    
    # 랜덤 샘플링
    random.seed(42)  # 재현 가능한 결과를 위해 시드 설정
    selected_pairs = random.sample(pair_dirs, min(num_samples, len(pair_dirs)))
    
    # pair 번호 추출
    selected_indices = []
    for pair_dir in selected_pairs:
        pair_num = int(pair_dir.split('_')[1])
        selected_indices.append(pair_num)
    
    selected_indices.sort()
    
    print(f"선정된 샘플: {selected_indices[:10]}... (총 {len(selected_indices)}개)")
    
    return selected_indices

def parse_label_file(filepath):
    """라벨 파일에서 대응점 좌표를 읽어옵니다."""
    try:
        points = []
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            # 형식: "1 296.0030653586301, 188.0888049593124"
                            if len(parts) >= 3:
                                x_str = parts[1].rstrip(',')  # 콤마 제거
                                y_str = parts[2]
                                x, y = float(x_str), float(y_str)
                                points.append([x, y])
                        except (ValueError, IndexError) as e:
                            continue
        return np.array(points) if points else None
    except Exception as e:
        return None

def get_gt_homography_from_labels(pair_idx, label_mapping, target_size, base_dir="."):
    """라벨 파일에서 GT 호모그래피를 계산합니다."""
    try:
        pair_key = str(pair_idx)
        if pair_key not in label_mapping:
            return None, None, None
        
        labels = label_mapping[pair_key]
        
        # 라벨 파일 경로 설정
        kompsat_filename = os.path.basename(labels['kompsat_label'])
        google_filename = os.path.basename(labels['google_label'])
        
        kompsat_label_path = os.path.join(base_dir, "labels", kompsat_filename)
        google_label_path = os.path.join(base_dir, "labels", google_filename)
        
        # 라벨 파일이 존재하는지 확인
        if not (os.path.exists(kompsat_label_path) and os.path.exists(google_label_path)):
            return None, None, None
        
        # 대응점 로드
        k_points = parse_label_file(kompsat_label_path)
        g_points = parse_label_file(google_label_path)
        
        if k_points is None or g_points is None:
            return None, None, None
        
        # 점 개수가 다른 경우 더 적은 개수에 맞춤
        min_points = min(len(k_points), len(g_points))
        if min_points < 4:
            return None, None, None
        
        # 더 적은 개수만큼만 사용
        k_points = k_points[:min_points]
        g_points = g_points[:min_points]
        
        # RANSAC을 사용해서 이상치 제거 및 호모그래피 추정
        best_H = None
        best_mask = None
        best_inliers = 0
        
        for threshold in [5.0, 8.0, 12.0]:
            H, mask = cv2.findHomography(g_points, k_points, cv2.RANSAC, threshold)
            if H is not None and mask is not None:
                inliers = np.sum(mask)
                if inliers > best_inliers and inliers >= 6:
                    best_H = H
                    best_mask = mask
                    best_inliers = inliers
        
        if best_H is None or best_inliers < 6:
            return None, None, None
        
        # RANSAC으로 선택된 inlier들만 사용
        inlier_indices = np.where(best_mask.ravel())[0]
        k_points_filtered = k_points[inlier_indices]
        g_points_filtered = g_points[inlier_indices]
        
        # 필터링된 점들로 최종 호모그래피 재계산
        H_gt, _ = cv2.findHomography(g_points_filtered, k_points_filtered, 0)
        
        if H_gt is None:
            return None, None, None
        
        # GT 좌표를 모델 입력 크기에 맞게 스케일링
        original_size = 1000
        scale_factor = target_size / original_size
        
        g_points_scaled = g_points_filtered * scale_factor
        k_points_scaled = k_points_filtered * scale_factor
        
        # 호모그래피도 스케일링에 맞게 조정
        H_gt_scaled = H_gt.copy()
        H_gt_scaled[0, 2] *= scale_factor
        H_gt_scaled[1, 2] *= scale_factor
            
        return H_gt_scaled, g_points_scaled, k_points_scaled
        
    except Exception as e:
        return None, None, None

def calculate_metrics(H_pred, gt_points_source, gt_points_target):
    """하나의 이미지 쌍에 대한 모든 평가지표를 계산합니다."""
    if H_pred is None or gt_points_source is None:
        return None

    # GT 좌표를 H_pred로 변환하여 예측 좌표 생성
    gt_source_h = np.hstack([gt_points_source, np.ones((len(gt_points_source), 1))])
    pred_target_h = (H_pred @ gt_source_h.T).T
    pred_target_points = pred_target_h[:, :2] / pred_target_h[:, 2:3]
    
    # 픽셀 오차 계산
    errors = np.linalg.norm(pred_target_points - gt_points_target, axis=1)
    
    # 평가지표 계산
    rmse = np.sqrt(np.mean(errors ** 2))
    
    pck_metrics = {}
    for t in Config.PCK_THRESHOLDS:
        pck_metrics[f'pck@{t}px'] = (errors < t).mean() * 100
    
    metrics = {
        'rmse': rmse,
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'std_error': np.std(errors),
        **pck_metrics,
    }
    return metrics

def evaluate_model(model, test_indices, label_mapping, device, model_name, base_dir="."):
    """모델을 평가합니다."""
    
    print(f"\n{'='*80}")
    print(f"🔄 {model_name} 평가 시작")
    print('='*80)
    
    model.eval()
    print(f"✅ {model_name} 모델 로드 완료")
    
    all_results = []
    
    for pair_idx in tqdm(test_indices, desc=f"{model_name} 평가"):
        try:
            # 이미지 로드 (절대 경로 사용)
            pair_dir = os.path.join(base_dir, 'processed_refine_data', f'pair_{pair_idx:04d}')
            img_source = cv2.cvtColor(cv2.imread(os.path.join(pair_dir, 'google.jpg')), cv2.COLOR_BGR2RGB)
            img_target = cv2.cvtColor(cv2.imread(os.path.join(pair_dir, 'kompsat.jpg')), cv2.COLOR_BGR2RGB)
            
            h, w = img_source.shape[:2]
            
            # GT Homography 및 GT 좌표 로드
            _, gt_source_pts, gt_target_pts = get_gt_homography_from_labels(pair_idx, label_mapping, Config.TARGET_SIZE, base_dir=base_dir)
            if gt_source_pts is None:
                continue

            # 모델 추론
            with torch.no_grad():
                warp, certainty = model.match(Image.fromarray(img_source), Image.fromarray(img_target), device=device)
                matches, _ = model.sample(warp, certainty)
            
            if matches is None or len(matches) == 0:
                continue
                
            kps_source, kps_target = model.to_pixel_coordinates(matches, h, w, h, w)
            kps_source = kps_source.cpu().numpy()
            kps_target = kps_target.cpu().numpy()

            if len(kps_source) < Config.MIN_MATCHES:
                continue
            
            # 예측 Homography 추정
            H_pred, _ = cv2.findHomography(kps_source, kps_target, cv2.RANSAC, Config.RANSAC_THRESHOLD)
            
            # 평가지표 계산
            metrics = calculate_metrics(H_pred, gt_source_pts, gt_target_pts)
            if metrics:
                metrics['model_name'] = model_name
                all_results.append(metrics)

        except Exception as e:
            if pair_idx < 10:
                print(f"  샘플 {pair_idx} 처리 실패: {e}")
            continue
            
    print(f"✅ {model_name} 평가 완료: {len(all_results)}개 샘플")
    return all_results

def print_comparison_results(original_results, dinov3_results):
    """비교 결과 출력"""
    
    print(f"\n{'='*80}")
    print("📊 원본 RoMa (DINOv2) vs DINOv3 RoMa 비교 결과")
    print('='*80)
    
    print(f"\nValid samples:")
    print(f"  원본 RoMa (DINOv2): {len(original_results)}")
    print(f"  DINOv3 RoMa: {len(dinov3_results)}")
    
    if len(original_results) == 0 and len(dinov3_results) == 0:
        print("\n⚠️  No valid results!")
        return
    
    # 각 모델별 통계 계산
    models_data = []
    
    for name, results in [
        ("원본 RoMa (DINOv2)", original_results),
        ("DINOv3 RoMa", dinov3_results)
    ]:
        if len(results) == 0:
            models_data.append({
                'name': name,
                'count': 0,
                'rmse_mean': None,
                'rmse_std': None,
                'rmse_median': None,
                'pck_1px': None,
                'pck_3px': None,
                'pck_5px': None,
            })
            continue
        
        # 통계 계산
        rmses = [r['rmse'] for r in results]
        pck_1px = [r['pck@1px'] for r in results]
        pck_3px = [r['pck@3px'] for r in results]
        pck_5px = [r['pck@5px'] for r in results]
        
        models_data.append({
            'name': name,
            'count': len(results),
            'rmse_mean': np.mean(rmses),
            'rmse_std': np.std(rmses),
            'rmse_median': np.median(rmses),
            'pck_1px': np.mean(pck_1px),
            'pck_3px': np.mean(pck_3px),
            'pck_5px': np.mean(pck_5px),
        })
    
    # 표 형식 출력
    print(f"\n{'='*80}")
    print("📊 성능 비교표 (Performance Comparison Table)")
    print('='*80)
    
    # 헤더
    print(f"\n{'Metric':<40} | {'원본 RoMa (DINOv2)':>20} | {'DINOv3 RoMa':>15}")
    print('-' * 85)
    
    # 샘플 수
    print(f"{'Valid Samples':<40} | {models_data[0]['count']:>20} | {models_data[1]['count']:>15}")
    print('-' * 85)
    
    # RMSE
    if models_data[0]['rmse_mean'] is not None and models_data[1]['rmse_mean'] is not None:
        print(f"{'RMSE Mean (px)':<40} | {models_data[0]['rmse_mean']:>19.2f} | {models_data[1]['rmse_mean']:>14.2f}")
        print(f"{'RMSE Std Dev (px)':<40} | {models_data[0]['rmse_std']:>19.2f} | {models_data[1]['rmse_std']:>14.2f}")
        print(f"{'RMSE Median (px)':<40} | {models_data[0]['rmse_median']:>19.2f} | {models_data[1]['rmse_median']:>14.2f}")
    else:
        print(f"{'RMSE Mean (px)':<40} | {'N/A':>20} | {'N/A':>15}")
    print('-' * 85)
    
    # PCK
    if models_data[0]['pck_1px'] is not None and models_data[1]['pck_1px'] is not None:
        print(f"{'PCK@1px (%)':<40} | {models_data[0]['pck_1px']:>19.2f} | {models_data[1]['pck_1px']:>14.2f}")
        print(f"{'PCK@3px (%)':<40} | {models_data[0]['pck_3px']:>19.2f} | {models_data[1]['pck_3px']:>14.2f}")
        print(f"{'PCK@5px (%)':<40} | {models_data[0]['pck_5px']:>19.2f} | {models_data[1]['pck_5px']:>14.2f}")
    else:
        print(f"{'PCK@1px (%)':<40} | {'N/A':>20} | {'N/A':>15}")
    
    print('='*85)
    
    # 성능 개선 요약
    if models_data[1]['rmse_mean'] is not None and models_data[0]['rmse_mean'] is not None:
        print(f"\n{'='*80}")
        print("📈 성능 개선 요약 (Performance Improvement Summary)")
        print('='*80)
        
        rmse_improvement = ((models_data[0]['rmse_mean'] - models_data[1]['rmse_mean']) / models_data[0]['rmse_mean']) * 100
        pck_1px_improvement = models_data[1]['pck_1px'] - models_data[0]['pck_1px']
        pck_3px_improvement = models_data[1]['pck_3px'] - models_data[0]['pck_3px']
        pck_5px_improvement = models_data[1]['pck_5px'] - models_data[0]['pck_5px']
        
        print(f"\n원본 RoMa (DINOv2) → DINOv3 RoMa:")
        print(f"  RMSE: {models_data[0]['rmse_mean']:.2f}px → {models_data[1]['rmse_mean']:.2f}px ({rmse_improvement:+.2f}%)")
        print(f"  PCK@1px: {models_data[0]['pck_1px']:.2f}% → {models_data[1]['pck_1px']:.2f}% ({pck_1px_improvement:+.2f}%p)")
        print(f"  PCK@3px: {models_data[0]['pck_3px']:.2f}% → {models_data[1]['pck_3px']:.2f}% ({pck_3px_improvement:+.2f}%p)")
        print(f"  PCK@5px: {models_data[0]['pck_5px']:.2f}% → {models_data[1]['pck_5px']:.2f}% ({pck_5px_improvement:+.2f}%p)")
        
        print('='*80)

# ============================================================================
# 메인 평가 함수 (Main Evaluation Function)
# ============================================================================
def compare_models():
    # --- 1. 초기 설정 ---
    print("=" * 80)
    print("🚀 원본 RoMa (DINOv2) vs DINOv3 RoMa 비교 평가를 시작합니다.")
    print(f"📊 랜덤 샘플 수: {Config.NUM_RANDOM_SAMPLES}개")
    print("=" * 80)
    
    # 다른 GPU 사용 (GPU 2번 사용 - 거의 비어있음)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 2:
            device = torch.device('cuda:2')  # GPU 2번 사용 (거의 비어있음)
            print(f"사용 장치: {device} (GPU 2번 - 거의 비어있음)")
        else:
            device = torch.device('cuda:0')
            print(f"사용 장치: {device} (GPU 0번)")
    else:
        device = torch.device('cpu')
        print(f"사용 장치: {device}")
    
    # --- 2. 랜덤 샘플 선정 ---
    print("\n[1/4] 랜덤 샘플 선정 중...")
    test_indices = get_random_samples(Config.DATA_DIR, Config.NUM_RANDOM_SAMPLES)
    
    # 라벨 매핑 로드
    label_mapping = load_json_file(Config.LABEL_MAPPING_FILE)
    
    print(f"✅ {len(test_indices)}개의 랜덤 테스트 샘플 선정 완료.")
    print(f"✅ {len(label_mapping)}개의 라벨 매핑 로드 완료.")
    
    # --- 3. 원본 RoMa 모델 평가 ---
    print("\n[2/4] 원본 RoMa (DINOv2) 평가 중...")
    
    # 원본 RoMa 디렉토리로 이동
    original_cwd = os.getcwd()
    original_dir = os.path.join(original_cwd, 'RoMa_original')
    os.chdir(original_dir)
    
    try:
        sys.path.insert(0, '.')
        from romatch import roma_outdoor
        
        original_model = roma_outdoor(device=device)
        original_model.load_state_dict(torch.load('./checkpoints/roma_outdoor.pth', map_location=device))
        
        original_results = evaluate_model(original_model, test_indices, label_mapping, device, "원본 RoMa (DINOv2)", original_cwd)
        
    finally:
        os.chdir(original_cwd)
    
    # --- 4. DINOv3 RoMa 모델 평가 ---
    print("\n[3/4] DINOv3 RoMa 평가 중...")
    
    # DINOv3 디렉토리로 이동
    dinov3_dir = os.path.join(original_cwd, 'RoMa_dinov3')
    os.chdir(dinov3_dir)
    
    try:
        # sys.path를 완전히 재설정하여 올바른 모듈 로드
        import importlib.util
        
        # 기존 romatch 모듈들 제거
        modules_to_remove = [key for key in sys.modules.keys() if key.startswith('romatch')]
        for module in modules_to_remove:
            del sys.modules[module]
        
        # sys.path 재설정
        sys.path = [p for p in sys.path if 'RoMa_original' not in p]
        sys.path.insert(0, dinov3_dir)
        
        # DINOv3 디렉토리의 roma_models.py 파일 직접 로드
        roma_models_path = os.path.join(dinov3_dir, 'romatch', 'models', 'model_zoo', 'roma_models.py')
        spec = importlib.util.spec_from_file_location("roma_models_dinov3", roma_models_path)
        roma_models_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(roma_models_module)
        
        roma_model = roma_models_module.roma_model
        
        dinov3_model = roma_model(
            resolution=(512, 512),  # DINOv3 패치 크기 16의 배수로 변경
            upsample_preds=True,
            device=device,
            weights=None,
            dinov3_model_name="facebook/dinov3-vit7b16-pretrain-sat493m",
            amp_dtype=torch.float32,
            original_roma_weights=os.path.join(original_cwd, 'RoMa_original', 'checkpoints', 'roma_outdoor.pth')
        )
        
        dinov3_results = evaluate_model(dinov3_model, test_indices, label_mapping, device, "DINOv3 RoMa", original_cwd)
        
    finally:
        os.chdir(original_cwd)
    
    # --- 5. 결과 비교 및 출력 ---
    print("\n[4/4] 결과 비교 및 분석 중...")
    print_comparison_results(original_results, dinov3_results)
    
    print("\n🎊 비교 평가 완료!")

if __name__ == "__main__":
    compare_models()

