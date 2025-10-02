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

# PSM 모드 리스트
psm_options = [3, 4, 6, 11]   # 문단형, 문장형, 블록형, 짧은 텍스트

# OEM 모드 리스트 (1=LSTM, 3=base)
oem_options = [1, 3]

for name, img in images:
    print("=" * 70)
    print(f" {name} 결과")

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE (대비 강화)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 4. Adaptive Threshold
    thresh = cv2.adaptiveThreshold(
        gray, #입력 이미지 = gray
        255,  #최대값으로 설정
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, #블록 사이즈
        10  # C
    )

    # OCR 실행 (PSM × OEM 조합)
    for psm in psm_options:
        for oem in oem_options:
            config = f'--oem {oem} --psm {psm} -c user_defined_dpi=300'

            text = pytesseract.image_to_string(thresh, lang="kor+eng", config=config)

            print("-" * 60)
            print(f" OEM {oem} | PSM {psm} 결과:")
            print(text.strip())

    print("\n")  # 이미지별 구분
