import cv2
import os
import glob


# ---------------------------------------------
# 画像の2値化を行う
# ---------------------------------------------
# フォルダ内のファイルを読み込む
for file_path in glob.glob("example/before/*"):
    # 画像が存在するかを確認
    if not os.path.exists(file_path):
        print(file_path + "は存在しません。")
        continue

    # 画像を読み込む
    img = cv2.imread(file_path)

    # グレースケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ガウシアンフィルターをかける
    gauss = cv2.GaussianBlur(gray, (9, 9), 0)

    # 2値化する
    thres = cv2.threshold(gauss, 180, 255, cv2.THRESH_BINARY)[1]

    # 画像を表示
    cv2.imwrite(file_path.replace("example/before/", "example/after/"), thres)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

