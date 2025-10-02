import pytesseract
import cv2

# Tesseract 실행 파일 경로
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# 이미지 경로
path_img1 = "./meme.jpg"
path_img2 = "./resume.png"
path_img3 = "./test.png"

images = [
    ("Image 1", cv2.imread(path_img1)),
    ("Image 2", cv2.imread(path_img2)),
    ("Image 3", cv2.imread(path_img3)),
]

psm_options = [6]   
oem_options = [3]   

# 전처리 함수
def preprocess_var(gray):
    variants = {}
    # 1. Gray
    variants["Gray"] = gray
    # 2. Otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants["Otsu"] = otsu
    # 3. Adaptive
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
    return variants

# 실행
for name, img in images:
    print("=" * 70)
    print(f" {name} 결과")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

        # 이미지 창으로 띄우기
        cv2.imshow(f"{name} - {variant_name}", proc_img)

    print("\n")

cv2.waitKey(0)   # 아무 키나 입력하면 닫히도록.
cv2.destroyAllWindows()
