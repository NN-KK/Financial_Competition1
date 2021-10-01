# Financial_Competition1
## 1. Data Processing
- Financial_Competition1ディレクトリで実行
    ```bash
    python Data/make_data.py
    ```
## 2. Prediction
- 以下を実行することにより, 
    ```bash
    python Model/lstm_with_gru/execute_updown.py
    ```
    ```bash
    python Model/lstm_with_gru/execute_price.py
    ```
- それぞれpredict_1_lstm_with_gru.csv, predict_2_lstm_with_gru.csvを予測ファイルとして取得
### parameter.csvは以下を実行することにより取得可能
- 29日の時点の最新データで実行して得たpatameter.csvを29, 30日ともに使用
- 実行するとそれぞれ3時間くらいかかります. 
    ```bash
    python Model/lstm_with_gru/fine_tuning_updown.py
    ```
    ```bash
    python Model/lstm_with_gru/fine_tuning_price.py
    ```