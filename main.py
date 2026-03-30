import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def main():
    # ---------------------------------------------------------------------
    # 1. 資料收集
    # ---------------------------------------------------------------------
    print("正在下載 S&P 500 股票數據...")
    ticker = '^GSPC'
    # 下載從 2021-01-01 到 2025-12-31 的資料
    df = yf.download(ticker, start='2021-01-01', end='2025-12-31')

    # 處理 yfinance 在提取單一標的時可能回傳的多重索引列 (MultiIndex columns)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ---------------------------------------------------------------------
    # 2. 資料預處理與特徵工程
    # ---------------------------------------------------------------------
    print("正在進行特徵工程...")
    
    # 填補缺失值 (使用前向填充 Forward Fill，確保無未來數據洩漏)
    df.ffill(inplace=True)

    # 基本特徵：計算移動平均線 (SMA)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()

    # 選配特徵：日報酬率 (Daily Return)
    df['Daily_Return'] = df['Close'].pct_change()
    
    # 選配特徵：波動率 (20日價格標準差)
    df['Volatility_20'] = df['Close'].rolling(window=20).std()

    # 標籤設定 (Labels) / 目標變數：預測隔日收盤價 (Next Day Close)
    # 將 Close 往上平移 1 天 (Shift(-1))，這樣對於時間 T，其 Target 就會是 T+1 的收盤價
    df['Target_Next_Close'] = df['Close'].shift(-1)

    # 清除因為計算移動平均線 (rolling) 以及標籤平移 (shift) 所產生的 NaN 值
    df.dropna(inplace=True)

    # 選擇最終要進入模型的特徵欄位以及預測標籤
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'SMA_60', 'Daily_Return', 'Volatility_20']
    X = df[features]
    y = df['Target_Next_Close']

    # ---------------------------------------------------------------------
    # 3. 資料集切割 (嚴格遵守時間序列，絕對禁止隨機洗牌 Shuffling)
    # ---------------------------------------------------------------------
    print("正在分割訓練集與測試集...")
    # Train: 2021-01-01 ~ 2024-12-31
    # Test: 2025-01-01 ~ 2025-12-31
    train_mask = (df.index >= '2021-01-01') & (df.index < '2025-01-01')
    test_mask = (df.index >= '2025-01-01') & (df.index <= '2025-12-31')

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    print(f"訓練集樣本數 (2021-2024): {len(X_train)}")
    print(f"測試集樣本數 (2025): {len(X_test)}")

    if len(X_test) == 0:
         print("⚠️ 警告：沒有找到2025年的測試集數據，請確認時間範圍區間！")
         return

    # ---------------------------------------------------------------------
    # 4. 模型建立與訓練
    # ---------------------------------------------------------------------
    print("正在訓練 Random Forest Regressor...")
    # 建立與訓練 Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    print("正在訓練 XGBoost Regressor...")
    # 建立與訓練 XGBoost
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)

    # ---------------------------------------------------------------------
    # 5. 預測與評估
    # ---------------------------------------------------------------------
    print("正在測試集上進行預測與評估...")
    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    # 計算 Mean Squared Error (MSE)
    rf_mse = mean_squared_error(y_test, rf_pred)
    xgb_mse = mean_squared_error(y_test, xgb_pred)

    print(f"\n【評估結果 (Mean Squared Error)】")
    print(f"Random Forest MSE: {rf_mse:,.2f}")
    print(f"XGBoost MSE:       {xgb_mse:,.2f}")

    # ---------------------------------------------------------------------
    # 6. 視覺化呈現
    # ---------------------------------------------------------------------
    print("\n正在生成並儲存視覺化圖表...")
    
    # a. 時間序列預測對比圖 (Prediction vs Actual)
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test.values, label='Actual S&P 500 Close', color='black', linewidth=2)
    plt.plot(y_test.index, rf_pred, label='Random Forest Prediction', color='blue', alpha=0.7)
    plt.plot(y_test.index, xgb_pred, label='XGBoost Prediction', color='red', alpha=0.7)
    plt.title('S&P 500 Next-Day Close Price Prediction (2025 Test Set)', fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=300)
    print("✅ 預測對比圖已儲存為 'prediction_comparison.png'")

    # b. 特徵重要性分析 (Feature Importance)
    rf_importance = rf_model.feature_importances_
    xgb_importance = xgb_model.feature_importances_

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Random Forest 長條圖
    axes[0].barh(features, rf_importance, color='blue', alpha=0.7)
    axes[0].set_title('Random Forest - Feature Importance')
    axes[0].set_xlabel('Importance')
    axes[0].invert_yaxis() # 讓數值最大的特徵顯示在上方

    # XGBoost 長條圖
    axes[1].barh(features, xgb_importance, color='red', alpha=0.7)
    axes[1].set_title('XGBoost - Feature Importance')
    axes[1].set_xlabel('Importance')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("✅ 特徵重要性圖表已儲存為 'feature_importance.png'")

    # 顯示圖表
    plt.show()

if __name__ == "__main__":
    main()
