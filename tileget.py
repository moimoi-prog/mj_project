import cv2
import os
import glob
import numpy as np
from pylsd.lsd import lsd
import math
from sympy import Symbol, solve


# 傾きと切片を求める
def calc_slope_intersept(x1, y1, x2, y2):
    a = (y1 - y2) / (x1 - x2)
    b = y1 - a * x1
    return (a, b)


# 直線の交点を求める(整数で返す)
def calc_closs(a1, b1, a2, b2):
    x = Symbol("x")
    y = Symbol("y")

    ex1 = a1 * x + b1 - y
    ex2 = a2 * x + b2 - y

    result = solve(((ex1, ex2)))

    return [round(result[x]), round(result[y])]


for file_path in glob.glob("img/before/*"):

    # 画像が存在するかを確認
    if not os.path.exists(file_path):
        print(file_path + "は存在しません。")
        continue

    # 画像を読み込む
    img = cv2.imread(file_path)

    # 画像の縦横の長さを取得
    height, width = img.shape[:2]

    # 画像の４隅の座標を取得(x, y)
    # 左上を原点とする
    # 左上
    p1 = (0, 0)
    # 右上
    p2 = (width, 0)
    # 左下
    p3 = (0, height)
    # 右下
    p4 = (width, height)

    # グレースケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ガウシアンフィルターをかける
    gauss = cv2.GaussianBlur(gray, (9, 9), 0)

    # 2値化する
    thres = cv2.threshold(gauss, 180, 255, cv2.THRESH_BINARY)[1]

    # 輪郭のみを検出する
    cons = cv2.findContours(thres,
                            cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_NONE)[0]

    if len(cons) > 0:
        # 輪郭を抽出できた場合の処理
        # 1つ目の領域を最大の面積とする
        max_con = cons[0]

        # 取り出した輪郭が複数の場合、絞り込みを行う
        if len(cons) > 1:
            for con in cons[1:]:
                # 一番大きいかを判定
                if cv2.contourArea(max_con) < cv2.contourArea(con):
                    max_con = con

        # 塗りつぶす(次の処理で牌の輪郭以外の直線を取得させないため)
        img_black = np.ones((height, width), np.uint8) * 0
        img_white = np.ones((height, width), np.uint8) * 255
        cv2.drawContours(img_black, [max_con], -1, color=(255, 255, 255), thickness=-1)

        # 牌の輪郭線はガタガタしているので、無理やりまっすぐにする(雰囲気で解読してください)
        # 線分を取得(斜め線は、だいたい途切れて取得される)
        linesL = lsd(img_black)

        # 座標のリスト(初期値は中心の座標)
        # ・０ーーーーーーーーーーーーーー２・
        # １　　　　　　　　　　　　　　　　３
        # ｜　　　　　　　　　　　　　　　　｜
        # ｜　　　　　　　　　　　　　　　　｜
        # ５　　　　　　　　　　　　　　　　７
        # ・４ーーーーーーーーーーーーーー６・
        cdn_list = [[width / 2, height / 2]] * 8

        # 斜めの場合は途切れているので、つなげる
        for line in linesL:
            # 線分の視点と終点の座標を取得()
            x1, y1, x2, y2 = map(int, line[:4])

            # 線分が縦線か横線かによって処理を分岐する
            if abs(x1 - x2) > abs(y1 - y2):
                # 横線の場合の処理
                # 0を取得
                cur = math.sqrt((p1[0] - cdn_list[0][0]) ** 2 + (p1[1] - cdn_list[0][1]) ** 2)
                candidate1 = math.sqrt((p1[0] - x1) ** 2 + (p1[1] - y1) ** 2)
                candidate2 = math.sqrt((p1[0] - x2) ** 2 + (p1[1] - y2) ** 2)

                if cur > candidate1:
                    cur = candidate1
                    cdn_list[0] = [x1, y1]

                if cur > candidate2:
                    cdn_list[0] = [x2, y2]

                # 2を取得
                cur = math.sqrt((p2[0] - cdn_list[2][0]) ** 2 + (p2[1] - cdn_list[2][1]) ** 2)
                candidate1 = math.sqrt((p2[0] - x1) ** 2 + (p2[1] - y1) ** 2)
                candidate2 = math.sqrt((p2[0] - x2) ** 2 + (p2[1] - y2) ** 2)

                if cur > candidate1:
                    cur = candidate1
                    cdn_list[2] = [x1, y1]

                if cur > candidate2:
                    cdn_list[2] = [x2, y2]

                # 4取得
                cur = math.sqrt((p3[0] - cdn_list[4][0]) ** 2 + (p3[1] - cdn_list[4][1]) ** 2)
                candidate1 = math.sqrt((p3[0] - x1) ** 2 + (p3[1] - y1) ** 2)
                candidate2 = math.sqrt((p3[0] - x2) ** 2 + (p3[1] - y2) ** 2)

                if cur > candidate1:
                    cur = candidate1
                    cdn_list[4] = [x1, y1]

                if cur > candidate2:
                    cdn_list[4] = [x2, y2]

                # 6を取得
                cur = math.sqrt((p4[0] - cdn_list[6][0]) ** 2 + (p4[1] - cdn_list[6][1]) ** 2)
                candidate1 = math.sqrt((p4[0] - x1) ** 2 + (p4[1] - y1) ** 2)
                candidate2 = math.sqrt((p4[0] - x2) ** 2 + (p4[1] - y2) ** 2)

                if cur > candidate1:
                    cur = candidate1
                    cdn_list[6] = [x1, y1]

                if cur > candidate2:
                    cdn_list[6] = [x2, y2]

            else:
                # 縦線の場合の処理
                # 1を取得
                cur = math.sqrt((p1[0] - cdn_list[1][0]) ** 2 + (p1[1] - cdn_list[1][1]) ** 2)
                candidate1 = math.sqrt((p1[0] - x1) ** 2 + (p1[1] - y1) ** 2)
                candidate2 = math.sqrt((p1[0] - x2) ** 2 + (p1[1] - y2) ** 2)

                if cur > candidate1:
                    cur = candidate1
                    cdn_list[1] = [x1, y1]

                if cur > candidate2:
                    cdn_list[1] = [x2, y2]

                # 3を取得
                cur = math.sqrt((p2[0] - cdn_list[3][0]) ** 2 + (p2[1] - cdn_list[3][1]) ** 2)
                candidate1 = math.sqrt((p2[0] - x1) ** 2 + (p2[1] - y1) ** 2)
                candidate2 = math.sqrt((p2[0] - x2) ** 2 + (p2[1] - y2) ** 2)

                if cur > candidate1:
                    cur = candidate1
                    cdn_list[3] = [x1, y1]

                if cur > candidate2:
                    cdn_list[3] = [x2, y2]

                # 5取得
                cur = math.sqrt((p3[0] - cdn_list[5][0]) ** 2 + (p3[1] - cdn_list[5][1]) ** 2)
                candidate1 = math.sqrt((p3[0] - x1) ** 2 + (p3[1] - y1) ** 2)
                candidate2 = math.sqrt((p3[0] - x2) ** 2 + (p3[1] - y2) ** 2)

                if cur > candidate1:
                    cur = candidate1
                    cdn_list[5] = [x1, y1]

                if cur > candidate2:
                    cdn_list[5] = [x2, y2]

                # 7を取得
                cur = math.sqrt((p4[0] - cdn_list[7][0]) ** 2 + (p4[1] - cdn_list[7][1]) ** 2)
                candidate1 = math.sqrt((p4[0] - x1) ** 2 + (p4[1] - y1) ** 2)
                candidate2 = math.sqrt((p4[0] - x2) ** 2 + (p4[1] - y2) ** 2)

                if cur > candidate1:
                    cur = candidate1
                    cdn_list[7] = [x1, y1]

                if cur > candidate2:
                    cdn_list[7] = [x2, y2]

        # 各辺の傾きと切片を求める
        # 上辺
        u_side = calc_slope_intersept(cdn_list[0][0], cdn_list[0][1], cdn_list[2][0], cdn_list[2][1])
        # 下辺
        d_side = calc_slope_intersept(cdn_list[4][0], cdn_list[4][1], cdn_list[6][0], cdn_list[6][1])
        # 左辺
        l_side = calc_slope_intersept(cdn_list[1][0], cdn_list[1][1], cdn_list[5][0], cdn_list[5][1])
        # 右辺
        r_side = calc_slope_intersept(cdn_list[3][0], cdn_list[3][1], cdn_list[7][0], cdn_list[7][1])

        # 辺の交点を求める
        # 上辺 + 左辺
        closs1 = calc_closs(u_side[0], u_side[1], l_side[0], l_side[1])
        # 上辺 + 右辺
        closs2 = calc_closs(u_side[0], u_side[1], r_side[0], r_side[1])
        # 下辺 + 左辺
        closs3 = calc_closs(d_side[0], d_side[1], l_side[0], l_side[1])
        # 下辺 + 右辺
        closs4 = calc_closs(d_side[0], d_side[1], r_side[0], r_side[1])

        # 射影変換しとく
        cdn_before = np.float32([closs1, closs2, closs3, closs4])
        cdn_after = np.float32([[0, 0], [1400, 0], [0, 130], [1400, 130]])

        # 射影変換
        ptf = cv2.getPerspectiveTransform(cdn_before, cdn_after)
        after = cv2.warpPerspective(img, ptf, (1400, 130))

        # 画像を分割して保存する
        for count in range(14):
            # 画像を保存
            clip = after[0:130, count * 100: (count + 1) * 100]
            cv2.imwrite("img/after/" + str(count) + ".png", clip)

        cv2.imwrite("img/after/after.png", after)

        # # 画像をファイルに出力
        # cv2.imshow(file_path + "img", img)
        # cv2.imshow(file_path + "after", after)

        # # 線分を引く
        # cv2.line(img, (cdn_list[0][0], cdn_list[0][1]), (cdn_list[2][0], cdn_list[2][1]), color=(0, 0, 255))
        # cv2.line(img, (cdn_list[4][0], cdn_list[4][1]), (cdn_list[6][0], cdn_list[6][1]), color=(0, 0, 255))
        # cv2.line(img, (cdn_list[1][0], cdn_list[1][1]), (cdn_list[5][0], cdn_list[5][1]), color=(0, 0, 255))
        # cv2.line(img, (cdn_list[3][0], cdn_list[3][1]), (cdn_list[7][0], cdn_list[7][1]), color=(0, 0, 255))

    else:
        # 輪郭を抽出できなかった場合の処理
        print('画像内に牌データは存在しませんでした。')

cv2.waitKey(0)
cv2.destroyAllWindows()
