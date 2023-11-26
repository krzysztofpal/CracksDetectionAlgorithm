import os
import cv2 as cv
from algorithm import compute

if __name__ == "__main__":
    img_path = os.path.join("data", "136TC-110-02(13;10)-5X-04.jpg")
    demo_path = "demo_results"

    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    result = compute(img, width=256, size_limit=8, output_plots_path=demo_path)


