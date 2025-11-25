import os
import cv2
import numpy as np
import json
import warnings
import logging
from paddleocr import PaddleOCR

# ===== 환경 설정 =====
warnings.filterwarnings("ignore")
logging.getLogger("ppocr").setLevel(logging.ERROR)
os.environ["QT_QPA_PLATFORM"] = "xcb"

# 복잡해서 실패했던 이미지 경로를 넣어주세요
IMAGE_PATH = "/home/intel/Documents/medibuddy/OCR/test_image/4.png" # 처방전 예시

# ==========================================
# 1. 좌표 기반 클러스터링 (핵심 로직 🔥)
# ==========================================
def cluster_text_boxes(bboxes, x_tol=20, y_tol=30):
    """
    글자 박스들을 받아 서로 가까운 것끼리 병합하여 큰 덩어리(문단)를 만듭니다.
    이미지 처리가 아닌, 순수 좌표 계산입니다.
    
    x_tol: 가로로 이만큼 떨어져 있어도 합침 (단어 사이 연결) - 작게 잡아야 단 분리됨
    y_tol: 세로로 이만큼 떨어져 있어도 합침 (줄 사이 연결)
    """
    if not bboxes: return []
    
    # [1] 초기화: 모든 글자 박스를 '클러스터 후보'로 등록
    # [x1, y1, x2, y2] 형태로 변환
    clusters = []
    for box in bboxes:
        pts = np.array(box, dtype=np.float32)
        if pts.size == 8: pts = pts.reshape(4, 2)
        x1 = np.min(pts[:, 0])
        x2 = np.max(pts[:, 0])
        y1 = np.min(pts[:, 1])
        y2 = np.max(pts[:, 1])
        clusters.append([x1, y1, x2, y2])

    # [2] 반복 병합 (더 이상 합쳐질 게 없을 때까지)
    changed = True
    while changed:
        changed = False
        new_clusters = []
        visited = [False] * len(clusters)
        
        for i in range(len(clusters)):
            if visited[i]: continue
            
            # 기준 박스
            base = clusters[i]
            visited[i] = True
            
            # 병합 루프 (기준 박스와 겹치거나 가까운 녀석들을 모두 흡수)
            merged_something = True
            while merged_something:
                merged_something = False
                for j in range(len(clusters)):
                    if visited[j]: continue
                    
                    target = clusters[j]
                    
                    # 거리 계산 (음수면 겹침, 양수면 떨어짐)
                    # 수평 거리: max(0, start2 - end1, start1 - end2)
                    dist_x = max(0, target[0] - base[2], base[0] - target[2])
                    
                    # 수직 거리
                    dist_y = max(0, target[1] - base[3], base[1] - target[3])
                    
                    # 조건: 가로/세로 거리가 허용치 이내인가?
                    if dist_x < x_tol and dist_y < y_tol:
                        # 병합 실행 (영역 확장)
                        base[0] = min(base[0], target[0]) # x1
                        base[1] = min(base[1], target[1]) # y1
                        base[2] = max(base[2], target[2]) # x2
                        base[3] = max(base[3], target[3]) # y2
                        
                        visited[j] = True
                        merged_something = True
                        changed = True # 한 번이라도 변했으면 전체 루프 다시
            
            new_clusters.append(base)
        
        clusters = new_clusters

    # [3] 정렬 (위->아래, 좌->우)
    # y좌표를 50px 단위로 퉁쳐서, 같은 줄에 있는 건 x좌표 순으로 정렬
    clusters.sort(key=lambda c: (int(c[1]/50), c[0]))
    
    return clusters

# ==========================================
# 2. 데이터 추출
# ==========================================
def extract_data_smart(result_item):
    extracted_data = []
    try:
        keys = []
        if hasattr(result_item, 'keys'): keys = list(result_item.keys())
        rec_texts = result_item['rec_texts'] if 'rec_texts' in keys else None
        dt_polys = result_item['dt_polys'] if 'dt_polys' in keys else None
        
        if rec_texts is None and 'res' in keys: return extract_data_smart(result_item['res'])
        if rec_texts is not None and dt_polys is not None:
            for i in range(len(rec_texts)):
                extracted_data.append({'text': rec_texts[i], 'bbox': dt_polys[i]})
            return extracted_data
    except: pass
    if isinstance(result_item, list):
         try:
             bbox = result_item[0]
             text_obj = result_item[1]
             text = text_obj[0] if isinstance(text_obj, (list, tuple)) else str(text_obj)
             return [{'text': text, 'bbox': bbox}]
         except: pass
    return []

def main():
    print("="*50)
    print(f"🧲 Coordinate Clustering OCR: {IMAGE_PATH}")
    print("="*50)

    if not os.path.exists(IMAGE_PATH):
        print("❌ 파일 없음")
        return

    image = cv2.imread(IMAGE_PATH)
    if image is None: return

    # 1. OCR 먼저 수행 (글자 위치를 알아야 묶으니까요)
    print("✓ 전체 텍스트 스캔 중...")
    ocr = PaddleOCR(lang='korean', use_angle_cls=True)
    result = ocr.ocr(image)

    # 2. 데이터 추출
    flat_data = []
    if isinstance(result, list):
        for item in result:
            data = extract_data_smart(item)
            if data: flat_data.extend(data)
            else:
                if isinstance(item, list):
                    for sub in item: flat_data.extend(extract_data_smart(sub))

    print(f"  >> 총 {len(flat_data)}개의 텍스트 조각 발견")
    
    if not flat_data:
        print("❌ 텍스트 없음")
        return

    # 3. [핵심] 글자 박스 좌표만 뽑아서 클러스터링
    print("✓ 좌표 기반 문단 응집 중...")
    
    # bbox만 리스트로 추출
    all_bboxes = [item['bbox'] for item in flat_data]
    
    # 🔥 파라미터 튜닝 가이드 🔥
    # x_tol=20: 단어 사이 간격 (이보다 멀면 다른 단/문단)
    # y_tol=30: 줄 간격 (이보다 멀면 다른 문단) -> 팍스로비드 같은 건 15~20 추천
    layout_clusters = cluster_text_boxes(all_bboxes, x_tol=30, y_tol=20)
    
    print(f"  >> 총 {len(layout_clusters)}개의 의미 덩어리(문단) 생성")

    # 4. 결과 매칭
    final_output = []
    for i, cluster in enumerate(layout_clusters):
        final_output.append({
            'id': i+1, 
            'bbox': cluster, # [x1, y1, x2, y2]
            'texts': []
        })

    # 텍스트 넣기
    for item in flat_data:
        text = item['text']
        bbox = item['bbox']
        
        pts = np.array(bbox, dtype=np.float32)
        if pts.size == 8: pts = pts.reshape(4, 2)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        
        # 클러스터 포함 여부
        for section in final_output:
            sx1, sy1, sx2, sy2 = section['bbox']
            # 약간의 오차 허용
            if sx1-5 <= cx <= sx2+5 and sy1-5 <= cy <= sy2+5:
                section['texts'].append({'text': text, 'cx': cx, 'cy': cy})
                break

    # 5. 저장 및 시각화
    json_data = []
    vis_img = image.copy()

    print("\n📜 [문단별 정리 결과]")
    for section in final_output:
        texts = section['texts']
        if not texts: continue
        
        # 시각화 (빨간 박스)
        sx1, sy1, sx2, sy2 = map(int, section['bbox'])
        cv2.rectangle(vis_img, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
        cv2.putText(vis_img, str(section['id']), (sx1, sy1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 내부 정렬 (줄->칸)
        texts.sort(key=lambda x: x['cy'])
        sorted_lines = []
        curr_line = [texts[0]]
        for i in range(1, len(texts)):
            if abs(texts[i]['cy'] - curr_line[-1]['cy']) < 15: # 같은 줄
                curr_line.append(texts[i])
            else:
                curr_line.sort(key=lambda x: x['cx'])
                sorted_lines.extend(curr_line)
                curr_line = [texts[i]]
        curr_line.sort(key=lambda x: x['cx'])
        sorted_lines.extend(curr_line)
        
        full_content = " ".join([t['text'] for t in sorted_lines])
        json_data.append({'block_id': section['id'], 'content': full_content})
        
        print(f"[Block {section['id']}] {full_content[:40]}...")

    filename = os.path.basename(IMAGE_PATH)
    json_path = f"final_cluster_coord_{filename}.json"
    vis_path = f"final_cluster_coord_vis_{filename}"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    cv2.imwrite(vis_path, vis_img)
    
    print("\n" + "="*50)
    print(f"💾 JSON 저장됨: {json_path}")
    print(f"📸 확인 이미지: {vis_path}")
    print("👉 팁: 만약 문단이 너무 잘게 쪼개지면 x_tol, y_tol 값을 조금만 늘려주세요!")

if __name__ == "__main__":
    main()