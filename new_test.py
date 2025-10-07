import pytesseract
import cv2
import numpy as np

# Tesseract 실행 파일 경로
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# 이미지 경로들
path_img1 = "./meme.jpg"
path_img2 = "./resume.png"
path_img3 = "./test.png"

images = [
    ("Image 1", cv2.imread(path_img1)),
    ("Image 2", cv2.imread(path_img2)),
    ("Image 3", cv2.imread(path_img3)),
]

# OCR 옵션
psm_options = [6]  
oem_options = [3]   

#함수 LIST
def resize_img(img, name, max_len=1000):
    h, w = img.shape[:2]
    print(f" {name} 원본 크기: {w} x {h}")
    if w > h:
        new_w = max_len
        new_h = int(h / w * max_len)
    else:
        new_h = max_len
        new_w = int(w / h * max_len)
    return cv2.resize(img, (new_w, new_h))

def morph_gradient(gray, ksize=(3,3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

def otsu_threshold(gray, invert=True):  #adaptive에서 otsu로 변경
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 흰색 비율 계산
    white_ratio = cv2.countNonZero(th) / float(th.size)

    # 흰 영역이 적을 경우 반전
    if white_ratio < 0.5:
        th = cv2.bitwise_not(th)

        # 반전 후 대비 손실 방지 -> gamma correction
        invGamma = 1.0 / 0.7   # 감마 보정 계수
        table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
        th = cv2.LUT(th, table)

        # 경계 선명화
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        th = cv2.filter2D(th, -1, kernel)

    return th

def clahe_contrast(gray):
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
    return clahe.apply(gray)

def morph_close(bin_img, ksize=(3, 3), iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed

#진단 함수
def quick_diagnose(gray, bin_img):
    contrast = float(gray.std())
    edges = cv2.Canny(bin_img, 50, 150)
    h, w = bin_img.shape[:2]
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 120, minLineLength=w//3, maxLineGap=10)
    line_count = 0 if lines is None else len(lines)
    fg_ratio = cv2.countNonZero(bin_img) / float(h*w)
    print(f"[diag] contrast={contrast:.1f}, line_count={line_count}, fg_ratio={fg_ratio:.3f}")
    return contrast, line_count, fg_ratio

def morph_topHat(gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

def morph_blackHat(gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

def remove_long_line(bin_img):
    edges = cv2.Canny(bin_img, 50, 150) # 윤곽선 감지
    lines = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi/180, threshold=120, #직선 감지 민감도
        minLineLength=bin_img.shape[1]//3, #최소 직선 길이 (이미지 가로변의 1/3)
        maxLineGap=10 #선 사이 최대 간격
    )
    cleaned = bin_img.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(cleaned, (x1,y1),(x2,y2), 0, 3) #선을 검은색으로 덮음
    return cleaned

def find_contour(bin_img):
    contours, _ = cv2.findContours(
        bin_img, 
        mode=cv2.RETR_EXTERNAL, #외곽선만 검출
        method=cv2.CHAIN_APPROX_SIMPLE  #꼭짓점만 저장
        )
    return contours

#자동 전처리 로직
def choose_preprocess(img, name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_tmp = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contrast, line_count, fg_ratio = quick_diagnose(gray, bin_tmp)

    print(f"{name} 진단 결과: contrast={contrast:.1f}, line_count={line_count}, fg_ratio={fg_ratio:.3f}")

    if contrast < 15:
        print("저대비: morph gradient, Otsu")
        gray = morph_topHat(gray)
        grad = morph_gradient(gray)
        th = otsu_threshold(grad)
        closed = morph_close(th)
        result = closed

    elif contrast > 35:
        print("고대비")
        if contrast > 45:
            gray = morph_blackHat(gray)
        th = otsu_threshold(gray)
        result = morph_close(th)

    else:
        print("중간 대비")
        grad = morph_gradient(gray)
        th = otsu_threshold(grad)
        result = morph_close(th)
    
    if line_count >= 3:
        print("표 또는 긴 선 감지")
        result = remove_long_line(result)

    return result  


# OCR & 출력
for name, img in images:
    if img is None:
        print(f"{name} 이미지를 불러올 수 없음.")
        continue

    print("=" * 70)
    print(f" {name} 결과")

    img_resized = resize_img(img, name)
    processed = choose_preprocess(img_resized, name)

    print("-" * 60)
    print(f"[OCR 결과: {name}]")
    for psm in psm_options:
        for oem in oem_options:
            config = f'--oem {oem} --psm {psm} -c user_defined_dpi=300'
            text = pytesseract.image_to_string(processed, lang="kor+eng", config=config)
            print(f" OEM {oem} | PSM {psm} 결과:")
            print(text.strip())

    cv2.imshow(f"{name} - Preprocessed", processed)
    print("\n")


cv2.waitKey(0)
cv2.destroyAllWindows()
