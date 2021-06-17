# -*- coding: utf-8 -*-
import numpy as np
from flask import Flask, render_template, request
import xgboost as xgb ## XGBoost 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
# 모델 불러오기
file_path = "C:\\Users\\pjhun\\jonan_project\\Flask Web Server\\model\\new_model.bst" # model path 설정!!
bst = xgb.Booster({'nthread': 4})
bst.load_model(file_path)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        # post 방식으로 데이터 변수 받아옴
        ton_temp = float(request.form.get('ton'))
        count_temp = int(request.form.get("ship"))
        sea_temp = str(request.form.get('feature1'))
        weather_temp = str(request.form.get('feature2'))
        accident_temp = str(request.form.get("feature3"))
        ship_temp = str(request.form.get("feature4"))

        # 더미화 변수로 예측하기 위한 전처리
        zo = pd.read_csv("C:\\Users\\pjhun\\jonan_project\\Flask Web Server\\data\\zonan_new.csv", encoding="CP949")
        zo["발생인원_범주형"] = zo["발생인원"]
        zo["발생인원_범주형"] = zo["발생인원_범주형"].apply(lambda x: 1 if x <= 5 else 2 if (x <= 20) else 3 if (x <= 50) else 4)
        col1 = ["발생해역", "기상상태", "발생유형", "선종", "사고선박수", "톤수"]
        col2 = ["발생인원_범주형"]
        col3 = ["발생해역", "기상상태", "발생유형", "선종", "사고선박수", "톤수", "발생인원_범주형"]

        zo_df = zo.loc[:, col3]

        zo_df["발생해역"] = zo_df["발생해역"].str.replace(' ', '')
        zo_df["발생해역"] = zo_df["발생해역"].str.replace("영해-EEZ", "배타적경제수역")
        zo_df["발생해역"] = zo_df["발생해역"].str.replace("EEZ-30마일이내", "경제수역30마일내")
        zo_df["발생해역"] = zo_df["발생해역"].str.replace("EEZ30마일이내", "경제수역30마일내")

        set(zo_df["발생해역"])

        def dummy_data(data, columns):
            for column in columns:
                data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
                data = data.drop(column, axis=1)
            return data

        dummy_columns = ["발생해역", "기상상태", "발생유형", "선종"]
        zo_df1 = dummy_data(zo_df, dummy_columns)

        zo_y = zo_df1["발생인원_범주형"]

        del zo_df1["발생인원_범주형"]
        zo_x = zo_df1

        x_train, x_test, y_train, y_test = train_test_split(zo_x, zo_y, train_size=0.7, test_size=0.3, random_state=0)
        test_data = x_test.drop(x_test.index)

        # 예측할 데이터 shape로 변환
        new_data = {'사고선박수': count_temp, '톤수': ton_temp, "발생해역_"+sea_temp:1,
                           "기상상태_"+weather_temp:1, "발생유형_"+accident_temp:1,"선종_"+ship_temp:1}
        test_data = test_data.append(new_data, ignore_index=True)
        test_data = test_data.fillna(0)
        xgtest = xgb.DMatrix(test_data)

        # 예측값을 찾아냅니다.
        ypred = bst.predict(xgtest)
        # 범주형 데이터로 전처리했던 부분 해당 범주에 맞는 발생인원으로 바꿈
        people = np.argmax(ypred)
        if people == 1:
            people = 5
        elif(people ==2):
            people = 20
        elif(people ==3):
            people =  50
        elif(people==4):
            people = 940

        return render_template('index.html', people=people)

if __name__ == '__main__':
   app.run(debug = True)