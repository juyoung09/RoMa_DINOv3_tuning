"""
================================================================================
Preprocessing Utilities for Satellite Image Matching
================================================================================

본 모듈은 위성 영상 정합을 위한 전처리 유틸리티 함수들을 제공합니다.

주요 기능:
================================================================================

1. **Label File Parsing**
   - 형식: "<point_id> <x>, <y>"
   - 중복 ID 처리: 평균값 사용
   - 파싱 오류 graceful handling

2. **Image Pair Grouping**
   - 복잡한 파일명 패턴 매칭
   - Google과 Kompsat 이미지 자동 페어링
   - Location 기반 유사도 계산

3. **Flow Map Generation**
   - Homography: 글로벌 변환 (강체 변환)
   - TPS (Thin Plate Spline): 부드러운 비선형 변형
   - RBF (Radial Basis Function): 국소적 변형
   - Auto selection: 포인트 개수 기반

4. **Point Validation**
   - 최소 개수 검증 (≥4)
   - Collinearity 검사
   - RANSAC outlier 제거

Transformation Methods 비교:
================================================================================

| Method     | Properties           | Use Case                  | Points  |
|------------|----------------------|---------------------------|---------|
| Homography | Global, Linear       | Planar scenes, rotation   | 4+      |
| TPS        | Global, Smooth       | Non-rigid deformation     | 10+     |
| RBF        | Local, Flexible      | Complex local changes     | 50+     |

Note: 위성 영상은 지형 변화가 있으므로 TPS/RBF가 더 적합할 수 있으나,
      sparse keypoint supervision으로 전환하면서 flow map 생성은 더 이상 사용되지 않음.

References:
================================================================================
[1] OpenCV Homography: https://docs.opencv.org/master/d9/dab/tutorial_homography.html
[2] TPS: Bookstein, "Principal Warps: Thin-Plate Splines...", PAMI 1989
[3] RBF: Buhmann, "Radial Basis Functions", Cambridge Univ. Press, 2003
"""
import os
import cv2
import numpy as np
import glob
from collections import defaultdict
from scipy.interpolate import Rbf


# ============================================================================
# Label File Parsing
# ============================================================================
def parse_label_file(label_path):
    """
    라벨 txt 파일에서 tie point 좌표 파싱
    
    File Format:
    ============================================================================
    각 라인은 다음 형식을 따름:
    <point_id> <x>, <y>
    
    Example:
    ```
    1 296.0030653586301, 188.0888049593124
    2 450.123456789012, 320.987654321098
    ...
    ```
    
    Parsing Rules:
    ============================================================================
    1. 공백으로 분리된 3개 필드 기대: ID, X좌표, Y좌표
    2. X좌표 뒤의 쉼표(,) 제거
    3. 중복 ID 발견 시 평균값 사용 (robust to annotation errors)
    4. 파싱 실패 라인은 조용히 무시 (ValueError, IndexError)
    
    Error Handling:
    ============================================================================
    - 빈 라인: 무시
    - 필드 개수 부족: 무시
    - 숫자 변환 실패: 무시
    - 중복 ID: 평균 좌표 계산
    
    Args:
        label_path (str): 라벨 파일 경로
    
    Returns:
        dict: {point_id (int): (x, y) (float tuple)}
            - point_id: tie point 고유 ID
            - (x, y): 픽셀 좌표 (원본 이미지 크기 기준)
    
    Example:
        >>> parse_label_file("label.txt")
        {1: (296.003, 188.089), 2: (450.123, 320.988), ...}
    """
    points_dict = {}
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                try:
                    point_id = int(parts[0])
                    x = float(parts[1].rstrip(','))
                    y = float(parts[2])
                    
                    # 중복 ID는 평균으로 처리
                    if point_id in points_dict:
                        old_x, old_y = points_dict[point_id]
                        points_dict[point_id] = ((old_x + x) / 2, (old_y + y) / 2)
                    else:
                        points_dict[point_id] = (x, y)
                except (ValueError, IndexError):
                    continue
    
    return points_dict


def extract_matching_points(kompsat_pts_dict, google_pts_dict):
    """
    두 딕셔너리에서 공통 point_id를 가진 매칭 포인트 추출
    
    Args:
        kompsat_pts_dict: {point_id: (x, y)}
        google_pts_dict: {point_id: (x, y)}
    
    Returns:
        kompsat_pts: (N, 2) numpy array
        google_pts: (N, 2) numpy array
        common_ids: 공통 ID 리스트
    """
    common_ids = sorted(set(kompsat_pts_dict.keys()) & set(google_pts_dict.keys()))
    
    if not common_ids:
        return None, None, []
    
    kompsat_pts = np.array([kompsat_pts_dict[pid] for pid in common_ids], dtype=np.float32)
    google_pts = np.array([google_pts_dict[pid] for pid in common_ids], dtype=np.float32)
    
    return kompsat_pts, google_pts, common_ids


# ============================================================================
# Image Pairing
# ============================================================================
def group_image_pairs(images_dir, labels_dir):
    """
    data_refine 폴더의 이미지들을 페어로 그룹핑
    
    패턴:
    - Kompsat: LCM00001_PS3_K3A_HEFEI_20190414.png
    - Google:  LCM00001_PS3_K3A_HEFEI_2019041_google.png
    
    Returns:
        List of dicts with keys: kompsat_img, google_img, kompsat_label, google_label
    """
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.png')))
    google_images = [f for f in image_files if '_google.png' in os.path.basename(f)]
    kompsat_images = [f for f in image_files if '_google.png' not in os.path.basename(f)]
    
    pairs = []
    
    for google_img in google_images:
        basename = os.path.basename(google_img)
        
        # Google 라벨
        google_label = os.path.join(labels_dir, basename.replace('.png', '.txt'))
        if not os.path.exists(google_label):
            continue
        
        # Kompsat 매칭
        google_prefix = basename.replace('_google.png', '')
        parts = google_prefix.split('_')
        
        if len(parts) < 3:
            continue
        
        # 공통 prefix (LCM번호_PS버전_K버전)
        common_prefix = '_'.join(parts[:3])
        location = parts[3] if len(parts) >= 4 else ""
        
        # Kompsat 후보 찾기
        kompsat_candidates = []
        for k_img in kompsat_images:
            k_basename = os.path.basename(k_img)
            
            if not k_basename.startswith(common_prefix):
                continue
            
            k_parts = k_basename.replace('.png', '').split('_')
            if len(k_parts) >= 4:
                k_location = k_parts[3]
                
                # Location 매칭 (부분 문자열 포함)
                if location.lower() in k_location.lower() or k_location.lower() in location.lower():
                    k_label = os.path.join(labels_dir, k_basename.replace('.png', '_k.txt'))
                    if os.path.exists(k_label):
                        kompsat_candidates.append((k_img, k_label))
        
        if not kompsat_candidates:
            continue
        
        # 가장 유사한 것 선택 (파일명 길이가 비슷한 것)
        kompsat_candidates.sort(
            key=lambda x: abs(len(google_prefix) - len(os.path.basename(x[0]).replace('.png', '')))
        )
        kompsat_img, kompsat_label = kompsat_candidates[0]
        
        pairs.append({
            'kompsat_img': kompsat_img,
            'google_img': google_img,
            'kompsat_label': kompsat_label,
            'google_label': google_label
        })
    
    return pairs


# ============================================================================
# Flow Map Generation
# ============================================================================
def validate_points(src_pts, dst_pts):
    """
    매칭 포인트 유효성 검증
    
    Returns:
        (is_valid, reason)
    """
    if src_pts is None or dst_pts is None:
        return False, "No matching points"
    
    if len(src_pts) == 0 or len(dst_pts) == 0:
        return False, "Empty point set"
    
    if len(src_pts) != len(dst_pts):
        return False, f"Count mismatch: {len(src_pts)} vs {len(dst_pts)}"
    
    if len(src_pts) < 4:
        return False, f"Too few points: {len(src_pts)} (need ≥4)"
    
    # Collinearity check
    if len(src_pts) >= 3:
        pts_centered = src_pts - src_pts.mean(axis=0)
        cov = np.cov(pts_centered.T)
        rank = np.linalg.matrix_rank(cov)
        if rank < 2:
            return False, "Points are collinear"
    
    return True, "OK"


def create_homography_flow(src_pts, dst_pts, image_shape):
    """
    Homography 기반 Dense Flow Map 생성
    
    Homography란?
    ============================================================================
    평면(planar) 객체의 3D 회전, 이동, 스케일을 표현하는 3×3 변환 행렬
    
    H: R³ → R³ (homogeneous coordinates)
    [x', y', 1]ᵀ = H · [x, y, 1]ᵀ
    
    where H = [h₁₁ h₁₂ h₁₃]
              [h₂₁ h₂₂ h₂₃]
              [h₃₁ h₃₂ h₃₃]
    
    8 자유도 (h₃₃=1로 정규화)
    
    Properties:
    ============================================================================
    - Global transformation (전체 이미지에 동일 적용)
    - Preserves straight lines (직선 보존)
    - 최소 4개 대응점 필요
    - RANSAC으로 outlier 제거 가능
    
    Limitations for Satellite Images:
    ============================================================================
    ❌ 지형 기복, 건물 높이 변화 표현 불가
    ❌ 국소적 비선형 변형 무시
    ✅ 하지만 간단하고 빠름, outlier에 robust (RANSAC)
    
    Algorithm:
    ============================================================================
    1. RANSAC으로 Homography 행렬 H 추정
       - Threshold: 5.0 pixels
       - Inlier mask 반환
    
    2. 전체 그리드 포인트 생성 (H×W)
    
    3. Homography 적용
       transformed = H @ [x, y, 1]ᵀ
    
    4. Flow 계산
       flow[y, x] = transformed[y, x] - [x, y]
    
    Args:
        src_pts (np.ndarray): Source keypoints (N, 2), pixel coords
        dst_pts (np.ndarray): Destination keypoints (N, 2), pixel coords
        image_shape (tuple): (height, width)
    
    Returns:
        np.ndarray: Dense flow map (H, W, 2), dtype=float32
            flow[y, x] = [dx, dy] displacement
    
    Note:
        만약 Homography 추정 실패 시 zero flow 반환
    """
    h, w = image_shape[:2]
    
    # RANSAC으로 outlier 제거하며 Homography 계산
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        return np.zeros((h, w, 2), dtype=np.float32)
    
    # 전체 그리드 생성
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    ones = np.ones_like(x_coords)
    grid = np.stack([x_coords, y_coords, ones], axis=-1).reshape(-1, 3)
    
    # Homography 적용
    transformed = (H @ grid.T).T
    transformed = transformed[:, :2] / transformed[:, 2:3]
    transformed = transformed.reshape(h, w, 2)
    
    # Flow 계산
    original = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)
    flow_map = transformed - original
    
    return flow_map.astype(np.float32)


def create_tps_flow(src_pts, dst_pts, image_shape):
    """
    Thin Plate Spline (TPS) 기반 Dense Flow Map 생성
    
    TPS란?
    ============================================================================
    물리학의 얇은 금속판 변형을 모델링한 보간 방법
    
    특성:
    - Globally smooth (전역적으로 부드러움)
    - Exact interpolation (control points에서 정확)
    - Minimizes bending energy (굽힘 에너지 최소화)
    
    Mathematical Formulation:
    ============================================================================
    f(x, y) = a₁ + a₂x + a₃y + Σᵢ wᵢ · U(||[x,y] - [xᵢ,yᵢ]||)
    
    where U(r) = r² log(r)  (radial basis function)
    
    Advantages:
    ============================================================================
    ✅ 부드러운 비선형 변형 표현
    ✅ Control points 정확 통과
    ✅ 지형 기복, 건물 변화 어느 정도 표현 가능
    
    Limitations:
    ============================================================================
    ❌ Global method → 먼 곳의 포인트도 영향
    ❌ 많은 포인트에서 계산 비용 증가
    ❌ Extrapolation 영역에서 불안정할 수 있음
    
    Use Case:
    ============================================================================
    - 중간 개수 포인트 (10~100개)
    - 전역적으로 일관된 변형 필요
    - 위성 영상의 기하 보정
    
    Args:
        src_pts (np.ndarray): Source keypoints (N, 2)
        dst_pts (np.ndarray): Destination keypoints (N, 2)
        image_shape (tuple): (height, width)
    
    Returns:
        np.ndarray: Dense flow map (H, W, 2), dtype=float32
    
    References:
        Bookstein, "Principal Warps: Thin-Plate Splines and the
        Decomposition of Deformations", PAMI 1989
    """
    h, w = image_shape[:2]
    
    # TPS 변환 계산
    tps = cv2.createThinPlateSplineShapeTransformer()
    
    src_pts_cv = src_pts.reshape(-1, 1, 2)
    dst_pts_cv = dst_pts.reshape(-1, 1, 2)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
    
    tps.estimateTransformation(dst_pts_cv, src_pts_cv, matches)
    
    # 전체 그리드에 적용
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    grid_points = np.stack([x_coords.ravel(), y_coords.ravel()], axis=-1).astype(np.float32)
    grid_points = grid_points.reshape(-1, 1, 2)
    
    transformed = tps.applyTransformation(grid_points)[1]
    transformed = transformed.reshape(h, w, 2)
    
    # Flow 계산
    original = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)
    flow_map = transformed - original
    
    return flow_map


def create_rbf_flow(src_pts, dst_pts, image_shape):
    """
    Radial Basis Function interpolation 기반 flow map 생성
    
    Args:
        src_pts: Source keypoints (N, 2)
        dst_pts: Destination keypoints (N, 2)
        image_shape: (height, width)
    
    Returns:
        flow_map: (H, W, 2)
    """
    h, w = image_shape[:2]
    
    # Flow vectors
    flow_vectors = dst_pts - src_pts
    
    # RBF interpolator (X, Y 각각)
    rbf_x = Rbf(src_pts[:, 0], src_pts[:, 1], flow_vectors[:, 0],
                function='thin_plate', smooth=0.1)
    rbf_y = Rbf(src_pts[:, 0], src_pts[:, 1], flow_vectors[:, 1],
                function='thin_plate', smooth=0.1)
    
    # 전체 그리드에 보간
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    flow_x = rbf_x(x_coords.ravel(), y_coords.ravel()).reshape(h, w)
    flow_y = rbf_y(x_coords.ravel(), y_coords.ravel()).reshape(h, w)
    
    flow_map = np.stack([flow_x, flow_y], axis=-1).astype(np.float32)
    
    return flow_map


def create_flow_map(src_pts, dst_pts, image_shape, method='auto'):
    """
    매칭 포인트로부터 dense flow map 생성
    
    Args:
        src_pts: Source keypoints (N, 2)
        dst_pts: Destination keypoints (N, 2)
        image_shape: (height, width)
        method: 'auto', 'homography', 'tps', 'rbf'
    
    Returns:
        flow_map: (H, W, 2)
        method_used: 실제 사용된 방법
    """
    # 자동 선택
    if method == 'auto':
        num_points = len(src_pts)
        if num_points >= 50:
            method = 'rbf'
        elif num_points >= 10:
            method = 'tps'
        else:
            method = 'homography'
    
    # Flow map 생성 (fallback 포함)
    try:
        if method == 'tps':
            flow_map = create_tps_flow(src_pts, dst_pts, image_shape)
            return flow_map, 'tps'
        elif method == 'rbf':
            flow_map = create_rbf_flow(src_pts, dst_pts, image_shape)
            return flow_map, 'rbf'
        elif method == 'homography':
            flow_map = create_homography_flow(src_pts, dst_pts, image_shape)
            return flow_map, 'homography'
    except Exception as e:
        # Fallback to homography
        if method != 'homography':
            flow_map = create_homography_flow(src_pts, dst_pts, image_shape)
            return flow_map, 'homography (fallback)'
        else:
            raise e


# ============================================================================
# Image Processing
# ============================================================================
def resize_image_and_points(image, points, target_size):
    """
    이미지와 매칭 포인트를 함께 리사이즈
    
    Args:
        image: 입력 이미지
        points: 매칭 포인트 (N, 2)
        target_size: (width, height)
    
    Returns:
        resized_image, scaled_points
    """
    h_orig, w_orig = image.shape[:2]
    
    # 이미지 리사이즈
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 포인트 스케일링
    scale_x = target_size[0] / w_orig
    scale_y = target_size[1] / h_orig
    scaled_points = points * [scale_x, scale_y]
    
    return resized, scaled_points


def save_processed_pair(output_dir, pair_idx, kompsat_img, google_img, flow_map):
    """
    전처리된 이미지 페어 저장
    
    Args:
        output_dir: 출력 디렉토리
        pair_idx: 페어 인덱스
        kompsat_img: Kompsat 이미지
        google_img: Google 이미지
        flow_map: Flow map (H, W, 2)
    """
    pair_dir = os.path.join(output_dir, f"pair_{pair_idx:04d}")
    os.makedirs(pair_dir, exist_ok=True)
    
    # 이미지 저장
    cv2.imwrite(
        os.path.join(pair_dir, 'kompsat.jpg'),
        kompsat_img,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    cv2.imwrite(
        os.path.join(pair_dir, 'google.jpg'),
        google_img,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    
    # Flow map 저장
    np.save(os.path.join(pair_dir, 'flow_map.npy'), flow_map)

