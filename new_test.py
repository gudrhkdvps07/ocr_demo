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


def adaptive_threshold(gray, block_size=25, C=10, invert=True):
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C
    )
    if invert:
        th = cv2.bitwise_not(th)
    return th

def morph_close(bin_img, ksize=(3, 3), iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed

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



# 전처리 실행
for name, img in images:
    if img is None:
        print(f"{name} 이미지를 불러올 수 없음.")
        continue

    print("=" * 70)
    print(f" {name} 결과")

    # 1. Resize
    img_resized = resize_img(img, name)

    # 2. Gray
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 3. Morph Gradient
    grad = morph_gradient(gray)

    # 4. Adaptive Threshold
    th = adaptive_threshold(grad, block_size=35, C=5)

    # 5. Morph close
    closed = morph_close(th, ksize=(3, 3), iterations=1)

    # 6. Long Line Remove
    cleaned = remove_long_line(closed)

    # 7. Contour
    contours = find_contour(closed)
    contour_img = np.zeros_like(closed)
    cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)

    # 전처리 관리
    variants = {
        "Gray": gray,
        "Morph Gradient": grad,
        "Adaptive Threshold": th,
        "Morph close" : closed,
        "Long Line Remove" : cleaned,
        "Find Contour" : contour_img,
    }


# OCR & 출력
    for variant_name, proc_img in variants.items():
        print("-" * 60)
        print(f"[전처리: {variant_name}]")

        for psm in psm_options:
            for oem in oem_options:
                config = f'--oem {oem} --psm {psm} -c user_defined_dpi=300'
                text = pytesseract.image_to_string(proc_img, lang="kor+eng", config=config)
                print(f" OEM {oem} | PSM {psm} 결과:")
                print(text.strip())
        
        
        pass_item = ["Gray", "Morph Gradient", "Adaptive Threshold"]

        if variant_name in pass_item:
            pass
        else:
            cv2.imshow(f"{name} - {variant_name}", proc_img)

    print("\n")

cv2.waitKey(0)
cv2.destroyAllWindows()
