import pytesseract
import cv2

# Tesseract 실행 파일 경로
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# 이미지 경로
path_img1 = "./meme.jpg"
path_img2 = "./resume.png"
path_img3 = "./test.png"

# 테스트할 이미지들을 리스트로 관리 (OpenCV로 불러옴)
images = [
    ("Image 1", cv2.imread(path_img1)),
    ("Image 2", cv2.imread(path_img2)),
    ("Image 3", cv2.imread(path_img3)),
]

# OCR 옵션
psm_options = [6]   
oem_options = [3]   

# 전처리 함수
def preprocess_var(gray):
    variants = {}

    # 1. Gray (순수 그레이스케일)
    variants["Gray"] = gray
    
    # 2. Otsu 이진화
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # 첫 번째 값은 무시, 두 번째 값만 otsu에 저장
    variants["Otsu"] = otsu
    
    # 3. Adaptive Threshold
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )
    variants["Adaptive"] = adaptive
    
    # 4. CLAHE + Adaptive
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    adaptive_clahe = cv2.adaptiveThreshold(
        clahe_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )
    variants["CLAHE+Adaptive"] = adaptive_clahe
    
    # 5. Gray + CLAHE + Morphology Closing + Adaptive Threshold
    clahe_img2 = clahe.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(clahe_img2, cv2.MORPH_CLOSE, kernel)
    adaptive_morph = cv2.adaptiveThreshold(
        morph, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )
    variants["CLAHE+Morph+Adaptive"] = adaptive_morph
    
    return variants

# 실행
for name, img in images:
    print("=" * 70)
    print(f" {name} 결과")

    #resize
    h, w = img.shape[:2]
    print(f" {name} 원본 크기: {w}, {h}")
    if w > h :
        new_w = 1000
        new_h = int((h / w) * 1000)
    else:
        new_h = 1000
        new_w = int((w / h) * 1000)
    
    img_resized = cv2.resize(img, (new_w, new_h))

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    variants = preprocess_var(gray)

    for variant_name, proc_img in variants.items():
        print("-" * 60)
        print(f"[전처리: {variant_name}]")

        # OCR 실행
        for psm in psm_options:
            for oem in oem_options:
                config = f'--oem {oem} --psm {psm} -c user_defined_dpi=300'
                text = pytesseract.image_to_string(proc_img, lang="kor+eng", config=config)
                print(f" OEM {oem} | PSM {psm} 결과:")
                print(text.strip())

        # 전처리된 이미지 창으로 띄우기
        cv2.imshow(f"{name} - {variant_name}", proc_img)

    print("\n")

cv2.waitKey(0)   # 아무 키나 입력하면 닫히도록
cv2.destroyAllWindows()
