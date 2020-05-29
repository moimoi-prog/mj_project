# -----------------------------------------------
# 必要なモジュールをインポート
# -----------------------------------------------
from symtable import Symbol

from PIL import Image
from flask import Flask, request, session, g, redirect, url_for, \
    abort, render_template, flash

import cv2
import os
import numpy as np
from pylsd.lsd import lsd
import math
from sympy import Symbol, solve


# -----------------------------------------------
# configを設定
# -----------------------------------------------
# DB接続用のデータを設定
from werkzeug.utils import secure_filename

USERNAME = "Evi"
PASSWORD = "evi0129"
MYSQL_HOST = "127.0.0.1"
MYSQL_USER = "root"
MYSQL_PASS = ""


# -----------------------------------------------
# メソッドを定義
# -----------------------------------------------
# インスタンスの生成
app = Flask(__name__)
app.secret_key = 'hogehoge'
PEOPLE_FOLDER = '/Users/moimoi_adm/PycharmProjects/mj_project/static/image'
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
app.config["CACHE_TYPE"] = "null"


# configの読み込み
app.config.from_object(__name__)


# URL = http://127.0.0.1:5000/
@app.route('/')
def main_page():
    title = "main_page"
    name = app.config['USERNAME']
    return render_template("index.html", title=title, name=name)


# URL = http://127.0.0.1:5000/login
@app.route('/login', methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form['username'] != app.config['USERNAME'] or request.form['password'] != app.config['PASSWORD']:
            error = 'Invalid username or password'
        else:
            session['logged_in'] = True
            flash('You were logged in')
            return redirect(url_for('main_page'))
    return render_template('login.html', error=error)


# URL = http://127.0.0.1:5000/logout
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You were logged out')
    return redirect(url_for('main_page'))


@app.route('/judge', methods=['GET', 'POST'])
def judge():
    if request.method == 'POST':
        img_file = request.files['img_file']
        if img_file:
            filename = secure_filename(img_file.filename)
            img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_file.save(img_url)
            result_str = judge_tile(img_url)
            img_url = "/static/image/" + filename

            return render_template('index.html', result_img=img_url, result_str=result_str)
        else:
            return ''' <p>許可されていない拡張子です</p> '''
    else:
        return redirect(url_for('index'))


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

def judge_tile(path):
    # 見本データを読み込む
    dic_sample_img = {}

    # 牌の種類と画像を紐づけた辞書を作成する
    # 明示的に紐づけるために、１つずつ格納する
    dic_sample_img.setdefault("一萬", cv2.cvtColor(cv2.imread("img/sample/m1.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("二萬", cv2.cvtColor(cv2.imread("img/sample/m2.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("三萬", cv2.cvtColor(cv2.imread("img/sample/m3.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("四萬", cv2.cvtColor(cv2.imread("img/sample/m4.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("五萬", cv2.cvtColor(cv2.imread("img/sample/m5.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("六萬", cv2.cvtColor(cv2.imread("img/sample/m6.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("七萬", cv2.cvtColor(cv2.imread("img/sample/m7.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("八萬", cv2.cvtColor(cv2.imread("img/sample/m8.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("九萬", cv2.cvtColor(cv2.imread("img/sample/m9.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("一筒", cv2.cvtColor(cv2.imread("img/sample/p1.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("二筒", cv2.cvtColor(cv2.imread("img/sample/p2.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("三筒", cv2.cvtColor(cv2.imread("img/sample/p3.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("四筒", cv2.cvtColor(cv2.imread("img/sample/p4.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("五筒", cv2.cvtColor(cv2.imread("img/sample/p5.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("六筒", cv2.cvtColor(cv2.imread("img/sample/p6.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("七筒", cv2.cvtColor(cv2.imread("img/sample/p7.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("八筒", cv2.cvtColor(cv2.imread("img/sample/p8.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("九筒", cv2.cvtColor(cv2.imread("img/sample/p9.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("一索", cv2.cvtColor(cv2.imread("img/sample/s1.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("二索", cv2.cvtColor(cv2.imread("img/sample/s2.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("三索", cv2.cvtColor(cv2.imread("img/sample/s3.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("四索", cv2.cvtColor(cv2.imread("img/sample/s4.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("五索", cv2.cvtColor(cv2.imread("img/sample/s5.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("六索", cv2.cvtColor(cv2.imread("img/sample/s6.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("七索", cv2.cvtColor(cv2.imread("img/sample/s7.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("八索", cv2.cvtColor(cv2.imread("img/sample/s8.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("九索", cv2.cvtColor(cv2.imread("img/sample/s9.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("北", cv2.cvtColor(cv2.imread("img/sample/north.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("南", cv2.cvtColor(cv2.imread("img/sample/south.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("東", cv2.cvtColor(cv2.imread("img/sample/east.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("西", cv2.cvtColor(cv2.imread("img/sample/west.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("白", cv2.cvtColor(cv2.imread("img/sample/haku.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("發", cv2.cvtColor(cv2.imread("img/sample/hatsu.png"), cv2.COLOR_BGR2GRAY))
    dic_sample_img.setdefault("中", cv2.cvtColor(cv2.imread("img/sample/chun.png"), cv2.COLOR_BGR2GRAY))

    # 画像のパスを定義
    input_path = "img/before/front.jpeg"

    # 画像が存在するかを確認
    if not os.path.exists(input_path):
        print(input_path + "は存在しません。")

    # 画像を読み込む
    img = cv2.imread(input_path)

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
        # 一番大きい部分を牌の塊と見なす
        if len(cons) > 1:
            for con in cons[1:]:
                # 一番大きいかを判定
                if cv2.contourArea(max_con) < cv2.contourArea(con):
                    max_con = con

        # 塗り潰した牌の塊の画像を生成する
        img_black = np.ones((height, width), np.uint8) * 0
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

        # --------------------------------------------------------
        # 牌の向きを取得
        # --------------------------------------------------------
        # 座標の生成
        cdn_before_img_black = np.float32([closs1, closs2, closs3, closs4])
        cdn_after_img_black = np.float32([[0, 0], [1400, 0], [0, 130], [1400, 130]])
        # 射影変換
        ptf_img_black = cv2.getPerspectiveTransform(cdn_before_img_black, cdn_after_img_black)
        after_img_black = cv2.warpPerspective(img_black, ptf_img_black, (1400, 130))

        after_thres = cv2.warpPerspective(thres, ptf_img_black, (1400, 130))

        str = ""

        # 牌の種類を判定する
        for count in range(14):
            # 牌を切り取る
            clip = after_thres[0:130, count * 100: (count + 1) * 100]

            # 判定する
            result_name = ""
            result_value = 0

            for sample_name, sample_img in dic_sample_img.items():
                matches = np.count_nonzero(clip == sample_img)
                if result_value < matches:
                    result_name = sample_name
                    result_value = matches

            str = str + result_name + ", "

        return str


# 主処理
if __name__ == "__main__":
    # 起動
    app.run(debug=True)
