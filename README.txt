# EE6550 2023 HW1

## To Execute
```
$ cd HW1_110061543
$ python3 HW1.py
```
## Dependencies installed
pandas
numpy
matplotlib
sklearn

## Question 1: Splitting wine.csv into training data and test data.
Wine.csv 中的資料已經根據 Type0, 1, 2 排序好了，所以直接根據 rows 分成 3 個
dataframes (type1, type2, type3)，並從三個 dataframes 中各隨機取 20 筆資料，將這
60 筆資料存成 test.csv，剩下的 423 資料存成 train.csv。

## Question 2: Evaluating the posterior probabilities and accuracy rate
第二部分要訓練 classifier 然後用 test data 來算準確率。首先用 pandas 將 train.csv 跟
test.csv 轉成 numpy arrays ， x_train 用來表示 feature values ，而 y_train 用來表示 labels
(x_test, y_test 亦同)。
