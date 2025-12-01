import os
import cv2
import numpy as np
import json
import warnings
import logging
from paddleocr import PaddleOCR
from openai import OpenAI

# ===== 환경 설정 =====
warnings.filterwarnings("ignore")
logging.getLogger("ppocr").setLevel(logging.ERROR)
os.environ["QT_QPA_PLATFORM"] = "xcb"

IMAGE_PATH = "/home/intel/Documents/medibuddy/OCR/OCR/medi-OCR/mediocr_test_result/Original_image/denpasa.png"

### ==========================================
### LLM SECTION: 클라이언트 설정
### ==========================================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ==========================================
# LLM: 문서 분류
# ==========================================
def classify_document(block_text):
    prompt = f"""
다음 텍스트가 어떤 종류의 의료 문서에 해당하는지 한 단어로만 답해줘.

- 약정보
- 복약지도서
- 입원안내서

텍스트:
{block_text}

출력 형식: 약정보 / 복약지도서 / 입원안내서 중 하나만
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content.strip()

# ==========================================
# LLM: 블록 요약
# ==========================================
def summarize_by_type(doc_type, block_text):
    if doc_type == "약정보":
        summary_prompt = f"""
다음 텍스트는 의약품 약정보입니다.
핵심만 3줄로 정리해주세요.

텍스트:
{block_text}
"""
    elif doc_type == "복약지도서":
        summary_prompt = f"""
다음 텍스트는 환자에게 제공되는 복약지도서입니다.
환자가 꼭 알아야 할 내용만 3줄로 요약해주세요.

텍스트:
{block_text}
"""
    else:
        summary_prompt = f"""
다음 텍스트는 입원안내서 관련 내용입니다.
환자의 입원 절차와 기본 안내를 중심으로 3줄 요약해주세요.

텍스트:
{block_text}
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": summary_prompt}]
    )
    return res.choices[0].message.content.strip()

# ==========================================
# 🔥 최종 문서 요약 (네 버전 그대로)
# ==========================================
def final_document_summary(block_result_json):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
당신은 의료 문서를 분석하는 시스템입니다.
아래는 OCR과 LLM을 통해 block 단위로 분석된 JSON입니다.
block_id 순서대로 문맥을 고려해 전체 문서를 통합 분석하고,
아래 형식으로만 답하십시오.

- 문서에 포함된 문서 타입 요약 (예: 약정보 4개, 복약지도서 2개)
- 전체 문서의 목적 요약
- 환자에게 꼭 필요한 핵심 정보 3가지
"""
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(block_result_json, ensure_ascii=False)}
        ]
    )
    return response.output_text

# ==========================================
# OCR 관련 헬퍼
# ==========================================
def cluster_text_boxes(bboxes, x_tol=20, y_tol=30):
    if not bboxes: return []
    clusters = []
    for box in bboxes:
        pts = np.array(box, dtype=np.float32)
        if pts.size == 8: pts = pts.reshape(4, 2)
        x1 = np.min(pts[:, 0]); x2 = np.max(pts[:, 0])
        y1 = np.min(pts[:, 1]); y2 = np.max(pts[:, 1])
        clusters.append([x1, y1, x2, y2])

    changed = True
    while changed:
        changed = False
        new_clusters = []
        visited = [False] * len(clusters)
        for i in range(len(clusters)):
            if visited[i]: continue
            base = clusters[i]
            visited[i] = True
            merged_something = True
            while merged_something:
                merged_something = False
                for j in range(len(clusters)):
                    if visited[j]: continue
                    target = clusters[j]
                    dist_x = max(0, target[0] - base[2], base[0] - target[2])
                    dist_y = max(0, target[1] - base[3], base[1] - target[3])
                    if dist_x < x_tol and dist_y < y_tol:
                        base[0] = min(base[0], target[0])
                        base[1] = min(base[1], target[1])
                        base[2] = max(base[2], target[2])
                        base[3] = max(base[3], target[3])
                        visited[j] = True
                        merged_something = True
                        changed = True
            new_clusters.append(base)
        clusters = new_clusters
    clusters.sort(key=lambda c: (int(c[1]/50), c[0]))
    return clusters

def extract_data_smart(result_item):
    extracted_data = []
    try:
        keys = list(result_item.keys()) if hasattr(result_item, 'keys') else []
        rec_texts = result_item['rec_texts'] if 'rec_texts' in keys else None
        dt_polys = result_item['dt_polys'] if 'dt_polys' in keys else None
        if rec_texts is None and 'res' in keys:
            return extract_data_smart(result_item['res'])
        if rec_texts is not None and dt_polys is not None:
            for i in range(len(rec_texts)):
                extracted_data.append({'text': rec_texts[i], 'bbox': dt_polys[i]})
            return extracted_data
    except:
        pass
    if isinstance(result_item, list):
        try:
            bbox = result_item[0]
            text_obj = result_item[1]
            text = text_obj[0] if isinstance(text_obj, (list, tuple)) else str(text_obj)
            return [{'text': text, 'bbox': bbox}]
        except:
            pass
    return []

# ==========================================
# 메인 파이프라인
# ==========================================
def main():
    print("="*50)
    print(f"OCR: {IMAGE_PATH}")
    print("="*50)

    if not os.path.exists(IMAGE_PATH):
        print("파일 없음")
        return

    image = cv2.imread(IMAGE_PATH)
    if image is None: return

    print("✓ OCR 실행 중...")
    ocr = PaddleOCR(lang='korean', use_angle_cls=True)
    result = ocr.ocr(image)

    flat_data = []
    if isinstance(result, list):
        for item in result:
            data = extract_data_smart(item)
            if data: flat_data.extend(data)
            else:
                if isinstance(item, list):
                    for sub in item:
                        flat_data.extend(extract_data_smart(sub))

    print(f"총 {len(flat_data)}개 텍스트")

    all_bboxes = [item['bbox'] for item in flat_data]
    layout_clusters = cluster_text_boxes(all_bboxes, x_tol=30, y_tol=20)

    final_output = []
    for i, cluster in enumerate(layout_clusters):
        final_output.append({
            'id': i+1,
            'bbox': cluster,
            'texts': []
        })

    for item in flat_data:
        text = item['text']
        bbox = item['bbox']
        pts = np.array(bbox, dtype=np.float32)
        if pts.size == 8: pts = pts.reshape(4, 2)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        for section in final_output:
            sx1, sy1, sx2, sy2 = section['bbox']
            if sx1-5 <= cx <= sx2+5 and sy1-5 <= cy <= sy2+5:
                section['texts'].append({'text': text, 'cx': cx, 'cy': cy})
                break

    json_data = []
    for section in final_output:
        texts = section['texts']
        if not texts: continue
        texts.sort(key=lambda x: x['cy'])
        sorted_lines = []
        curr_line = [texts[0]]
        for i in range(1, len(texts)):
            if abs(texts[i]['cy'] - curr_line[-1]['cy']) < 15:
                curr_line.append(texts[i])
            else:
                curr_line.sort(key=lambda x: x['cx'])
                sorted_lines.extend(curr_line)
                curr_line = [texts[i]]
        curr_line.sort(key=lambda x: x['cx'])
        sorted_lines.extend(curr_line)
        full_content = " ".join([t['text'] for t in sorted_lines])
        json_data.append({'block_id': section['id'], 'content': full_content})

    # ==========================================
    # 🔥 LLM 파이프라인 적용
    # ==========================================
    llm_results = []
    print("\n=== LLM 파이프라인 작동 ===")
    for block in json_data:
        block_text = block["content"]
        doc_type = classify_document(block_text)
        summary = summarize_by_type(doc_type, block_text)
        llm_results.append({
            "block_id": block["block_id"],
            "document_type": doc_type,
            "summary": summary
        })
        print(f"\n[Block {block['block_id']}]")
        print(f"- 문서타입: {doc_type}")
        print(f"- 요약: {summary}")

    with open("llm_output.json", "w", encoding="utf-8") as f:
        json.dump(llm_results, f, ensure_ascii=False, indent=4)

    print("\n완료! → llm_output.json 저장")

    # ==========================================
    # 🔥 최종 문서 요약
    # ==========================================
    print("\n=== 최종 문서 요약 생성 ===")
    summary_text = final_document_summary(llm_results)
    print("\n===== 최종 요약 =====")
    print(summary_text)

    with open("final_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("\n완료! → final_summary.txt 저장")

if __name__ == "__main__":
    main()