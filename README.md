# Kaggle M5 Forecasting - Accuracy

## 競賽問題

**目的**

- 替 42840 組商品銷售量的時間序列提供最準確地預測方法

**內容**

- 層級的單位銷售數據，從產品商店級別，可再依據產品部門，產品類別，商店和州等類別匯聚成 12 個層級
- 除銷售數據外，另提供解釋性變量（售價、特殊事件、星期等）

**參考網址**

- https://www.kaggle.com/c/web-traffic-time-series-forecasting

## 程式解說

### 檔案

- feature_engineering/features_extract.py
- feature_engineering/validation_data_process.py.py
- feature_engineering/evaluation_data_process.py
- model_train.py

### 執行流程

1. 特徵處理

   ```bash
   python feature_engineering/validation_data_process.py.py
   python feature_engineering/evaluation_data_process.py
   ```

   針對 validation 與 evaluation data 做數據處理與特徵工程，其中內容包含：

   - 刪除未上式商品的無效銷售數據： `del_unlist_product_sales`
   - 萃取**價格**相關特徵：`extract_prices_features`
   - 萃取**時間**相關特徵：`extract_datetime_features`
   - 萃取**銷售**相關特徵：`extract_sales_features`
   - 類別特徵轉換：`transform_category_features`

   最終將處理好的資料以 pickle 的方式儲存，另外針對 evaluation data 需另外儲存 validation 期間的 true label，以利線下評估模型的預測效度

   

2. 模型訓練

   ```bash
   python model_train.py
   ```

   模型：

   - LightGBM

   訓練方式：

   - 針對各商品類別與各店面獨立建立模型，並為了減少記憶體用量，各次訓練所需的資料會分批讀取
   - 每個模型訓練完後，直接進行 validation 與 evaluation data 的預測

   產出：

   - 合併各模型所產出的預測結果，並以所提供的 `submission.csv` 的格式輸出