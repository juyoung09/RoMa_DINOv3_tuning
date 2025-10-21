
"""
범용 위성 영상 정합 비교 평가 스크립트 (Universal Satellite Image Registration Comparison Evaluation Script)

⭐ 이 스크립트의 목적:
   - 원본 RoMa (DINOv2) vs DINOv3 RoMa를 일관된 기준으로 비교 평가
   - 모델의 실용적인 최종 성능과 상세한 진단 정보를 동시에 제공

⭐ 평가 철학:
   - 실제 사용 시나리오를 모방하여, 모델이 예측한 특징점으로 Homography를 추정하고,
     이 Homography가 정답(GT) 좌표를 얼마나 정확하게 변환하는지 측정.

⭐ 주요 평가지표:
   1. RMSE (Root Mean Square Error): 최종 변환의 평균적인 기하학적 정확도 (핵심 성능 지표)
   2. Success Rate @ 5, 10px: 전체 중 정합 성공(RMSE < 임계값) 비율 (모델의 안정성/실용성 지표)
   3. PCK @ 1, 3, 5px: 최종 변환 오차 기반의 정확도 (정밀도 지표)
   4. Transformation Errors: 변환 행렬을 회전, 크기, 이동 오차로 분해하여 상세 진단
"""

import os
import sys
import json
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
try:
    import pandas as pd
except ImportError:
    print("pandas가 설치되지 않았습니다. pip install pandas를 실행하세요.")
    exit(1)

# ============================================================================
# 설정 (Configuration)
# ============================================================================
class Config:
    # --- 데이터 및 모델 경로 ---
    DATA_DIR = "./processed_refine_data"  # 전처리된 데이터 폴더
    TEST_SAMPLES_FILE = "./test_samples_10.json"  # 평가할 샘플 목록
    LABEL_MAPPING_FILE = os.path.join(DATA_DIR, "label_mapping.json")
    
    # 비교할 모델들 (별도 폴더에서 로드)
    ORIGINAL_ROMA_DIR = "./RoMa_original"
    DINOV3_ROMA_DIR = "./RoMa_dinov3"
    
    # --- 평가 설정 ---
    TARGET_SIZE = 560  # 이미지를 리사이즈할 크기
    RANSAC_THRESHOLD = 3.0  # RANSAC 이상치 제거 임계값
    MIN_MATCHES = 8  # Homography 추정을 위한 최소 매칭 수

    # --- 평가지표 임계값 ---
    PCK_THRESHOLDS = [1, 3, 5]  # PCK 계산을 위한 픽셀 임계값
    SUCCESS_THRESHOLDS = [5, 10] # Success Rate 계산을 위한 RMSE 임계값


# ============================================================================
# 유틸리티 함수 (Utility Functions)
# ============================================================================
def load_json_file(filepath):
    """JSON 파일을 로드합니다."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)

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
                            # parts[1] = "296.0030653586301," parts[2] = "188.0888049593124"
                            if len(parts) >= 3:
                                x_str = parts[1].rstrip(',')  # 콤마 제거
                                y_str = parts[2]
                                x, y = float(x_str), float(y_str)
                                points.append([x, y])
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Skipping line {line_num} in {filepath}: {line} (error: {e})")
                            continue
        return np.array(points) if points else None
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def get_gt_homography_from_labels(pair_idx, label_mapping, target_size, base_dir="."):
    """라벨 파일에서 GT 호모그래피를 계산합니다."""
    try:
        pair_key = str(pair_idx)
        if pair_key not in label_mapping:
            return None, None, None
        
        labels = label_mapping[pair_key]
        
        # 라벨 파일 경로를 현재 폴더의 labels 디렉토리 기준으로 수정
        kompsat_filename = os.path.basename(labels['kompsat_label'])
        google_filename = os.path.basename(labels['google_label'])
        
        # base_dir을 사용하여 절대 경로 설정 (os.chdir 때문에 상대 경로가 문제됨)
        kompsat_label_path = os.path.join(base_dir, "labels", kompsat_filename)
        google_label_path = os.path.join(base_dir, "labels", google_filename)
        
        # 라벨 파일이 존재하는지 확인
        if not (os.path.exists(kompsat_label_path) and os.path.exists(google_label_path)):
            if pair_idx < 3:  # 처음 3개만 디버깅 출력
                print(f"  샘플 {pair_idx}: 라벨 파일 경로 확인")
                print(f"    Kompsat: {kompsat_label_path} (존재: {os.path.exists(kompsat_label_path)})")
                print(f"    Google: {google_label_path} (존재: {os.path.exists(google_label_path)})")
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
        # 여러 번 시도해서 가장 좋은 결과 선택
        best_H = None
        best_mask = None
        best_inliers = 0
        
        for threshold in [5.0, 8.0, 12.0]:  # 더 관대한 임계값 시도
            H, mask = cv2.findHomography(g_points, k_points, cv2.RANSAC, threshold)
            if H is not None and mask is not None:
                inliers = np.sum(mask)
                if inliers > best_inliers and inliers >= 6:  # 최소 6개 이상의 inlier 필요
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
        
        # GT 좌표를 모델 입력 크기(560x560)에 맞게 스케일링
        # 원본 이미지 크기를 가정 (일반적으로 1000x1000 정도)
        original_size = 1000  # 원본 이미지 크기 추정
        scale_factor = target_size / original_size
        
        g_points_scaled = g_points_filtered * scale_factor
        k_points_scaled = k_points_filtered * scale_factor
        
        # 호모그래피도 스케일링에 맞게 조정
        H_gt_scaled = H_gt.copy()
        H_gt_scaled[0, 2] *= scale_factor  # tx
        H_gt_scaled[1, 2] *= scale_factor  # ty
            
        return H_gt_scaled, g_points_scaled, k_points_scaled
        
    except Exception as e:
        return None, None, None

def decompose_homography(H):
    """Homography 행렬을 이동, 회전, 크기 성분으로 분해합니다."""
    if H is None:
        return np.nan, np.nan, np.nan

    tx, ty = H[0, 2], H[1, 2]
    translation = np.sqrt(tx**2 + ty**2)
    
    a, b, c, d = H[0, 0], H[0, 1], H[1, 0], H[1, 1]
    
    # 회전각 (atan2 사용)
    rotation = np.arctan2(c, a) * (180 / np.pi)
    
    # 크기 (Scale)
    scale = np.sqrt(a**2 + c**2)
    
    return translation, rotation, scale

def calculate_metrics(H_pred, gt_points_source, gt_points_target):
    """
    하나의 이미지 쌍에 대한 모든 평가지표를 계산합니다.
    H_pred: 모델이 예측한 Homography
    gt_points_source: 변환의 기준이 되는 GT 좌표 (예: Google 이미지 GT 좌표)
    gt_points_target: 변환의 목표가 되는 GT 좌표 (예: Kompsat 이미지 GT 좌표)
    """
    if H_pred is None or gt_points_source is None:
        return None

    # 1. GT 좌표를 H_pred로 변환하여 예측 좌표 생성
    gt_source_h = np.hstack([gt_points_source, np.ones((len(gt_points_source), 1))])
    pred_target_h = (H_pred @ gt_source_h.T).T
    pred_target_points = pred_target_h[:, :2] / pred_target_h[:, 2:3]
    
    # 2. 픽셀 오차(error) 계산
    errors = np.linalg.norm(pred_target_points - gt_points_target, axis=1)
    
    # 3. 평가지표 계산
    rmse = np.sqrt(np.mean(errors ** 2))
    
    pck_metrics = {}
    for t in Config.PCK_THRESHOLDS:
        pck_metrics[f'pck@{t}px'] = (errors < t).mean() * 100
        
    trans_err, rot_err, scale_err = decompose_homography(H_pred)
    
    metrics = {
        'rmse': rmse,
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'std_error': np.std(errors),
        **pck_metrics,
        'translation_error': trans_err,
        'rotation_error': rot_err,
        'scale_error': abs(scale_err - 1.0) # 스케일은 1이 기준이므로 1과의 차이
    }
    return metrics

def evaluate_model_in_directory(model_dir, model_name, test_indices, label_mapping, device):
    """특정 디렉토리에서 모델을 로드하고 평가"""
    
    print(f"\n{'='*80}")
    print(f"🔄 {model_name} 평가 시작")
    print('='*80)
    
    # 해당 디렉토리로 이동
    original_cwd = os.getcwd()
    os.chdir(model_dir)
    
    try:
        # 해당 디렉토리의 모델 import
        sys.path.insert(0, '.')
        from romatch import roma_outdoor
        
        # 모델 로드
        print(f"{model_name} 모델 로드 중...")
        model = roma_outdoor(device=device)
        
        # 가중치 로드 (원본 RoMa의 경우)
        if "original" in model_name.lower():
            model.load_state_dict(torch.load('./checkpoints/roma_outdoor.pth', map_location=device))
        
        model.eval()
        print(f"✅ {model_name} 모델 로드 완료")
        
        all_results = []
        
        for pair_idx in tqdm(test_indices, desc=f"{model_name} 평가"):
            try:
                # 이미지 로드
                pair_dir = os.path.join('../processed_refine_data', f'pair_{pair_idx:04d}')
                img_source = cv2.cvtColor(cv2.imread(os.path.join(pair_dir, 'google.jpg')), cv2.COLOR_BGR2RGB)
                img_target = cv2.cvtColor(cv2.imread(os.path.join(pair_dir, 'kompsat.jpg')), cv2.COLOR_BGR2RGB)
                
                h, w = img_source.shape[:2]
                
                # GT Homography 및 GT 좌표 로드 (라벨 파일 사용)
                _, gt_source_pts, gt_target_pts = get_gt_homography_from_labels(pair_idx, label_mapping, Config.TARGET_SIZE, base_dir=original_cwd)
                if gt_source_pts is None:
                    if pair_idx < 3:  # 처음 3개만 에러 출력
                        print(f"  샘플 {pair_idx}: GT 호모그래피 로드 실패")
                    continue

                # 모델 추론 (RoMa 기준)
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
                
                # 예측 Homography(H_pred) 추정
                H_pred, _ = cv2.findHomography(kps_source, kps_target, cv2.RANSAC, Config.RANSAC_THRESHOLD)
                
                # 평가지표 계산
                metrics = calculate_metrics(H_pred, gt_source_pts, gt_target_pts)
                if metrics:
                    metrics['model_name'] = model_name
                    all_results.append(metrics)

            except Exception as e:
                if pair_idx < 10:  # 처음 10개 에러 출력
                    print(f"  샘플 {pair_idx} 처리 실패: {e}")
                    import traceback
                    traceback.print_exc()
                continue
                
        print(f"✅ {model_name} 평가 완료: {len(all_results)}개 샘플")
        return all_results
        
    finally:
        os.chdir(original_cwd)

# ============================================================================
# 메인 평가 함수 (Main Evaluation Function)
# ============================================================================
def compare_models():
    # --- 1. 초기 설정 ---
    print("=" * 80)
    print("🚀 원본 RoMa (DINOv2) vs DINOv3 RoMa 비교 평가를 시작합니다.")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")
    
    # --- 2. 데이터 로드 ---
    print("\n[1/4] 데이터 로드 중...")
    test_samples_data = load_json_file(Config.TEST_SAMPLES_FILE)
    test_indices = test_samples_data.get('selected_indices') or [s['pair_idx'] for s in test_samples_data.get('samples', [])]
    
    # 라벨 매핑 로드
    label_mapping = load_json_file(Config.LABEL_MAPPING_FILE)
    
    print(f"✅ {len(test_indices)}개의 테스트 샘플 로드 완료.")
    print(f"✅ {len(label_mapping)}개의 라벨 매핑 로드 완료.")
    
    # --- 3. 모델별 평가 ---
    print("\n[2/4] 모델별 평가 진행 중...")
    
    original_results = evaluate_model_in_directory(
        Config.ORIGINAL_ROMA_DIR, 
        "원본 RoMa (DINOv2)", 
        test_indices, 
        label_mapping,
        device
    )
    
    dinov3_results = evaluate_model_in_directory(
        Config.DINOV3_ROMA_DIR, 
        "DINOv3 RoMa", 
        test_indices, 
        label_mapping,
        device
    )
    
    # --- 4. 결과 비교 및 출력 ---
    print("\n[3/4] 결과 비교 및 분석 중...")
    print_comparison_results(original_results, dinov3_results)
    
    print("\n[4/4] 비교 평가 완료!")

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
                'translation_error': None,
                'rotation_error': None,
                'scale_error': None,
            })
            continue
        
        # 통계 계산
        rmses = [r['rmse'] for r in results]
        pck_1px = [r['pck@1px'] for r in results]
        pck_3px = [r['pck@3px'] for r in results]
        pck_5px = [r['pck@5px'] for r in results]
        translation_errors = [r['translation_error'] for r in results]
        rotation_errors = [r['rotation_error'] for r in results]
        scale_errors = [r['scale_error'] for r in results]
        
        models_data.append({
            'name': name,
            'count': len(results),
            'rmse_mean': np.mean(rmses),
            'rmse_std': np.std(rmses),
            'rmse_median': np.median(rmses),
            'pck_1px': np.mean(pck_1px),
            'pck_3px': np.mean(pck_3px),
            'pck_5px': np.mean(pck_5px),
            'translation_error': np.mean(translation_errors),
            'rotation_error': np.mean(rotation_errors),
            'scale_error': np.mean(scale_errors),
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
    print(f"{'RMSE Mean (px)':<40} | {models_data[0]['rmse_mean']:>19.2f} | {models_data[1]['rmse_mean']:>14.2f}" if models_data[1]['rmse_mean'] else f"{'RMSE Mean (px)':<40} | {models_data[0]['rmse_mean']:>19.2f} | {'N/A':>15}")
    print(f"{'RMSE Std Dev (px)':<40} | {models_data[0]['rmse_std']:>19.2f} | {models_data[1]['rmse_std']:>14.2f}" if models_data[1]['rmse_std'] else f"{'RMSE Std Dev (px)':<40} | {models_data[0]['rmse_std']:>19.2f} | {'N/A':>15}")
    print(f"{'RMSE Median (px)':<40} | {models_data[0]['rmse_median']:>19.2f} | {models_data[1]['rmse_median']:>14.2f}" if models_data[1]['rmse_median'] else f"{'RMSE Median (px)':<40} | {models_data[0]['rmse_median']:>19.2f} | {'N/A':>15}")
    print('-' * 85)
    
    # PCK
    print(f"{'PCK@1px (%)':<40} | {models_data[0]['pck_1px']:>19.2f} | {models_data[1]['pck_1px']:>14.2f}" if models_data[1]['pck_1px'] else f"{'PCK@1px (%)':<40} | {models_data[0]['pck_1px']:>19.2f} | {'N/A':>15}")
    print(f"{'PCK@3px (%)':<40} | {models_data[0]['pck_3px']:>19.2f} | {models_data[1]['pck_3px']:>14.2f}" if models_data[1]['pck_3px'] else f"{'PCK@3px (%)':<40} | {models_data[0]['pck_3px']:>19.2f} | {'N/A':>15}")
    print(f"{'PCK@5px (%)':<40} | {models_data[0]['pck_5px']:>19.2f} | {models_data[1]['pck_5px']:>14.2f}" if models_data[1]['pck_5px'] else f"{'PCK@5px (%)':<40} | {models_data[0]['pck_5px']:>19.2f} | {'N/A':>15}")
    print('-' * 85)
    
    # 기하학적 오차
    print(f"{'Translation Error (px)':<40} | {models_data[0]['translation_error']:>19.2f} | {models_data[1]['translation_error']:>14.2f}" if models_data[1]['translation_error'] else f"{'Translation Error (px)':<40} | {models_data[0]['translation_error']:>19.2f} | {'N/A':>15}")
    print(f"{'Rotation Error (deg)':<40} | {models_data[0]['rotation_error']:>19.2f} | {models_data[1]['rotation_error']:>14.2f}" if models_data[1]['rotation_error'] else f"{'Rotation Error (deg)':<40} | {models_data[0]['rotation_error']:>19.2f} | {'N/A':>15}")
    print(f"{'Scale Error':<40} | {models_data[0]['scale_error']:>19.2f} | {models_data[1]['scale_error']:>14.2f}" if models_data[1]['scale_error'] else f"{'Scale Error':<40} | {models_data[0]['scale_error']:>19.2f} | {'N/A':>15}")
    
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


if __name__ == "__main__":
    # --- 사용 예시 ---
    # 터미널에서 `python compare_original_vs_dinov3.py`를 실행합니다.
    compare_models()