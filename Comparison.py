import cv2
import numpy as np

# 画像を比較し、一致していない箇所を表示する

# 元画像
img_original = cv2.imread("example/after/after1.png")

# 比較対象画像１枚目(同じ画像)
img_comparison1 = cv2.imread("example/after/after1.png")

# 比較対象画像２枚目(違う画像)
img_comparison2 = cv2.imread("example/after/after3.png")

# 画像が完全一致するかを判定する
print(np.array_equal(img_original, img_comparison1))
print(np.array_equal(img_original, img_comparison2))

# 画素がどのくらい一致しているかを確認する
print(np.count_nonzero(img_original == img_comparison1))
print(np.count_nonzero(img_original == img_comparison2))
