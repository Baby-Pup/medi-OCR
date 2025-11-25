#지정된 이미지에서 전체 내용 OCR


import os
# 리눅스 환경에서 Qt 플랫폼 플러그인 오류 방지
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import json

# ===== 이미지 경로 =====
IMAGE_PATH = "/home/intel/Documents/medibuddy/OCR/OCR/medi-OCR/baseocr_test_result/Original_image/denpasa.png"  # <-- 분석할 이미지 파일명 또는 경로
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# 분석 설정
CONFIDENCE_THRESHOLD = 0.3  # 신뢰도 임계값
DETECT_BRAND_NAME = True    # 가장 큰 글씨(약품명) 찾기 모드
CENTER_WEIGHT = 0.3         # 중앙 가중치 (클수록 중앙에 있는 글씨 선호)
DEBUG_MODE = True          # 디버그 로그 출력 여부

# ===== 유틸리티 함수들 =====

def extract_ocr_data_from_result(result):
    """OCR 결과에서 데이터 추출"""
    ocr_data = []
    if not result: return ocr_data

    for res in result:
        # PaddleOCR 결과 구조 대응 (Dictionary 또는 Object)
        dt_polys = res.get('dt_polys') if isinstance(res, dict) else getattr(res, 'dt_polys', None)
        rec_texts = res.get('rec_texts') if isinstance(res, dict) else getattr(res, 'rec_texts', None)
        rec_scores = res.get('rec_scores') if isinstance(res, dict) else getattr(res, 'rec_scores', None)

        if dt_polys is not None and rec_texts is not None:
            # numpy 변환
            if hasattr(dt_polys, 'tolist'): dt_polys = dt_polys.tolist()
            if hasattr(rec_scores, 'tolist'): rec_scores = rec_scores.tolist()
            
            for bbox, text, score in zip(dt_polys, rec_texts, rec_scores):
                if score >= CONFIDENCE_THRESHOLD:
                    ocr_data.append({'bbox': bbox, 'text': text, 'score': float(score)})
    return ocr_data

def calculate_bbox_area(bbox):
    bbox = np.array(bbox)
    return (np.max(bbox[:, 0]) - np.min(bbox[:, 0])) * (np.max(bbox[:, 1]) - np.min(bbox[:, 1]))

def get_bbox_center(bbox):
    bbox = np.array(bbox)
    return np.mean(bbox[:, 0]), np.mean(bbox[:, 1])

def find_brand_name(ocr_data, image_shape):
    """
    [수정판 v3]
    - 1글자 노이즈 제거
    - 위치 상관없이 '글자 크기(높이)' 가중치 극대화
    """
    if not ocr_data: return None
    
    # 이미지 전체 크기
    img_h, img_w = image_shape[:2]
    
    max_score = -1
    brand_info = None
    
    for item in ocr_data:
        bbox = np.array(item['bbox'])
        text = item['text']
        
        # 1. [필터] 1글자는 과감하게 버림 (노이즈 제거)
        # 단, 글자 크기가 이미지 높이의 1/5 이상으로 엄청 크면 봐줌 (한 글자 약 이름일 수도 있으니)
        box_height = np.max(bbox[:, 1]) - np.min(bbox[:, 1])
        if len(text) < 2 and box_height < (img_h * 0.2):
            continue
            
        # 2. [필터] 너무 긴 문장 제거 (15자 이상)
        if len(text) > 15:
            continue
            
        # 3. [점수 계산] 로직 변경
        width = np.max(bbox[:, 0]) - np.min(bbox[:, 0])
        height = box_height
        area = width * height

        
        # 점수 = 높이의 제곱 (크기가 클수록 점수가 기하급수적으로 커짐)
        score = height * height
        
        # [보너스] '제품명' 같은 단어가 포함되어 있으면 가산점 살짝
        if "제품명" in text:
             score *= 1.2

        if score > max_score:
            max_score = score
            brand_info = item.copy()
            brand_info.update({
                'ranking_score': score
            })
    
    return brand_info

def draw_ocr_on_image(image, ocr_data, brand_info=None):
    """이미지에 박스와 텍스트 그리기 (폰트 크기 축소판)"""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    h, w = image.shape[:2]
    
    # [수정] 폰트 크기 대폭 축소
    # 기존 0.03(3%) -> 0.015(1.5%)로 줄임. 최소 사이즈도 10으로 줄임.
    base_font_size = max(int(h * 0.005), 12)
    # 브랜드 폰트는 조금 더 크게
    brand_font_size = max(int(h * 0.03), 20)
    
    try:
        font = ImageFont.truetype(FONT_PATH, base_font_size)
        brand_font = ImageFont.truetype(FONT_PATH, brand_font_size)
    except:
        font = ImageFont.load_default()
        brand_font = ImageFont.load_default()

    brand_bbox = brand_info['bbox'] if brand_info else None

    for item in ocr_data:
        bbox = [tuple(p) for p in item['bbox']]
        text = item['text']
        
        # 브랜드 여부 확인 (Numpy 배열 비교)
        is_brand = False
        if brand_bbox is not None:
            is_brand = np.array_equal(np.array(item['bbox']), np.array(brand_bbox))
        
        # 색상 및 스타일 설정
        if is_brand:
            color = (0, 0, 255) # 파란색
            width = 5
            current_font = brand_font
        else:
            color = (0, 0, 0) # 빨간색
            width = 2
            current_font = font
        
        # 박스 그리기
        draw.polygon(bbox, outline=color, width=width)
        
        # 텍스트 그리기 (박스 위쪽)
        x, y = item['bbox'][0]
        
        # [수정] 텍스트가 너무 빽빽하면 보기 싫으니까
        # 일반 텍스트는 박스 바로 위에 작게 그림
        text_x = x
        text_y = y - current_font.size - 2
        
        # 텍스트 배경 (가독성 위해)
        try:
            left, top, right, bottom = draw.textbbox((text_x, text_y), text, font=current_font)
            # 배경 박스도 살짝 여유 있게
            draw.rectangle((left-2, top-2, right+2, bottom+2), fill=color)
        except:
            pass
            
        draw.text((text_x, text_y), text, font=current_font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def main():
    print("=" * 50)
    print(f"📷 이미지 OCR 분석 시작: {IMAGE_PATH}")
    print("=" * 50)

    # 1. 이미지 로드
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 오류: 파일이 존재하지 않습니다 -> {IMAGE_PATH}")
        return

    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("❌ 오류: 이미지를 읽을 수 없습니다. 파일 형식을 확인하세요.")
        return
        
    print(f"✓ 이미지 로드 성공 ({image.shape[1]}x{image.shape[0]})")

    # 2. PaddleOCR 초기화
    print("✓ PaddleOCR 엔진 초기화 중...")
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        lang='korean',
        device='cpu',     # GPU 사용시 'gpu'로 변경
        # show_log=False    # 불필요한 로그 숨김
    )

    # 3. 예측 실행
    print("✓ OCR 분석 수행 중...")
    try:
        # PaddleOCR에 이미지 경로 대신 numpy 배열(image)을 직접 넘겨도 됨
        result = ocr.predict(input=image)
        print(result)
        
    except Exception as e:
        print(f"❌ OCR 실행 중 오류 발생: {e}")
        return

    # 4. 데이터 가공
    ocr_data = extract_ocr_data_from_result(result)
    print(f"✓ 텍스트 검출 완료: 총 {len(ocr_data)}개 항목")

    # 5. 주요 정보(약품명) 찾기
    brand_info = None
    if DETECT_BRAND_NAME:
        brand_info = find_brand_name(ocr_data, image.shape)
    
    # 6. 결과 출력 (콘솔)
    print("\n" + "-" * 30)
    print("[ 분석 결과 리포트 ]")
    if brand_info:
        print(f"💊 추정 약품명 (Main): {brand_info['text']}")
        print(f"   - 신뢰도: {brand_info['score']:.2f}")
        print(f"   - 위치점수: {brand_info['ranking_score']:.0f}")
    else:
        print("💊 추정 약품명: 감지되지 않음")
    
    print("\n📜 전체 텍스트 목록:")
    for i, item in enumerate(ocr_data, 1):
        prefix = ">>" if brand_info and item['text'] == brand_info['text'] else f"{i:02d}"
        print(f" {prefix} [{item['score']:.2f}] {item['text']}")
    print("-" * 30)

    # 7. 결과 시각화 및 저장
    final_image = draw_ocr_on_image(image, ocr_data, brand_info)
    
    output_filename = f"result_{os.path.basename(IMAGE_PATH)}"
    cv2.imwrite(output_filename, final_image)
    print(f"\n💾 결과 이미지 저장됨: {output_filename}")

    # 8. 화면 표시
    cv2.imshow(f"OCR Result - {os.path.basename(IMAGE_PATH)}", final_image)
    
    print("\n키보드 아무 키나 누르면 종료됩니다...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()