# 決策樹演算法實作與分析

**課程作業：資料探勘-決策樹分類器專題**

---

## 📋 目錄

1. [專案概述](#專案概述)
2. [專案架構](#專案架構)
3. [環境需求](#環境需求)
4. [資料集說明](#資料集說明)
5. [程式檔案說明](#程式檔案說明)
6. [執行方法](#執行方法)
7. [演算法說明](#演算法說明)
8. [結果分析](#結果分析)
9. [注意事項](#注意事項)

---

## 專案概述

專案包含三個主要部分：

- **Part 1**：使用 scikit-learn 建立 baseline 模型
- **Part 2**：手動實作 ID3、C4.5、CART、C5.0 四種決策樹演算法
- **Part 3**：CART 決策樹的成本複雜度剪枝（Cost Complexity Pruning）分析

---

## 專案架構

```
hw1_m11435021/
│
├── data/                               # 資料集目錄
│   ├── adult.data                      # 訓練資料
│   └── adult.test                      # 測試資料
│
├── src/                                # 原始碼目錄
│   ├── hw1_part1_baseline.ipynb        # Part 1: Baseline 模型
│   ├── hw1_part2_preprocessing.py      # Part 2: 統一資料前處理模組
│   ├── hw1_part2_id3_implementation.py # Part 2: ID3 決策樹實作
│   ├── hw1_part2_C45_implementation.py # Part 2: C4.5 決策樹實作
│   ├── hw1_part2_cart_implementation.py # Part 2: CART 決策樹實作
│   ├── hw1_part2_C50_implementation.py # Part 2: C5.0 決策樹實作
│   └── hw1_part3_cart_pruning.py       # Part 3: CART 剪枝分析
│
├── results/                            # 結果輸出目錄（執行後產生）
├── logs/                               # 日誌檔案目錄（執行後產生）
└── README.md                           # 本說明文件
```

---

## 環境需求

### Python 版本
- Python 3.8 或以上

### 必要套件

```bash
# 核心套件
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0

# C5.0 需要的額外套件
rpy2>=3.4.0  # Python 與 R 的介面
```

### R 套件（僅 C5.0 需要）

如果要執行 C5.0 演算法，需要安裝 R 語言及以下套件：

```r
# 在 R console 中執行
install.packages("C50")
```

### 安裝方式

```bash
# 使用 pip 安裝 Python 套件
pip install pandas numpy scikit-learn matplotlib

# 如果需要執行 C5.0
pip install rpy2
```

---

## 資料集說明

### Adult Income Dataset

本專案使用 UCI Machine Learning Repository 的 Adult Income 資料集。

- **來源**：[UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **任務**：預測個人年收入是否超過 $50K
- **特徵數量**：14 個特徵（6 個連續型、8 個類別型）
- **訓練集大小**：32,561 筆（清理後約 30,162 筆）
- **測試集大小**：16,281 筆（清理後約 15,060 筆）
> [!WARNING]  
> 需留意 ' ?' 缺失值的格式變體會導致清除或者讀取問題。

### 特徵說明

| 特徵名稱 | 類型 | 說明 |
|---------|------|------|
| age | 連續 | 年齡 |
| workclass | 類別 | 工作類型 |
| fnlwgt | 連續 | 最終權重 |
| education | 類別 | 教育程度 |
| education-num | 連續 | 教育年數 |
| marital-status | 類別 | 婚姻狀態 |
| occupation | 類別 | 職業 |
| relationship | 類別 | 家庭關係 |
| race | 類別 | 種族 |
| sex | 類別 | 性別 |
| capital-gain | 連續 | 資本利得 |
| capital-loss | 連續 | 資本損失 |
| hours-per-week | 連續 | 每週工作時數 |
| native-country | 類別 | 國籍 |
| **income** | **目標** | **<=50K 或 >50K** |

---

## 程式檔案說明

### Part 1: Baseline 模型

#### `hw1_part1_baseline.ipynb`

- **功能**：使用 scikit-learn 的 `DecisionTreeClassifier` 建立基準模型
- **特色**：
  - 資料載入與清理
  - 類別特徵 One-Hot 編碼
  - 訓練 CART 決策樹
  - 計算訓練集與測試集準確率
- **執行方式**：使用 Jupyter Notebook 開啟並執行

---

### Part 2: 決策樹演算法實作

#### `hw1_part2_preprocessing.py` - 統一資料前處理模組

- **功能**：提供 Part2 演算法統一的資料前處理介面
- **核心類別**：`UnifiedDataPreprocessor`
- **主要功能**：
  - 自動載入訓練與測試資料
  - 處理缺失值（刪除含 `?` 的資料）
  - 類別特徵編碼（使用 `LabelEncoder`）
  - 連續特徵離散化（可選，ID3 需要）
  - 訓練集/驗證集分割（可選，用於剪枝）
- **使用範例**：
  ```python
  from hw1_part2_preprocessing import UnifiedDataPreprocessor

  preprocessor = UnifiedDataPreprocessor(data_dir='../data')
  X_train, X_test, y_train, y_test = preprocessor.get_processed_data(
      discretize=False,  # 是否離散化
      validation_split=0.0  # 驗證集比例
  )
  ```

---

#### `hw1_part2_id3_implementation.py` - ID3 決策樹

- **演算法特色**：
  - 使用資訊增益（Information Gain）作為分裂標準
  - 使用熵（Entropy）計算不純度
  - 僅支援離散特徵（連續特徵需先離散化）
  - 每個特徵只能使用一次
- **核心類別**：`ID3DecisionTree`
- **主要功能**：
  - `fit(X, y)`：訓練模型
  - `predict(X)`：預測
  - `prune(X_val, y_val)`：後剪枝
  - `print_tree()`：顯示樹結構
- **執行方式**：
  ```bash
  cd src
  python hw1_part2_id3_implementation.py
  ```
- **輸出**：
  - 訓練/驗證/測試集準確率
  - 混淆矩陣與分類報告
  - 決策樹統計資訊（節點數、深度）
  - 儲存模型檔案 `id3_model.pkl`

---

#### `hw1_part2_C45_implementation.py` - C4.5 決策樹

- **演算法特色**：
  - 使用增益比率（Gain Ratio）作為分裂標準（改進 ID3 的偏向問題）
  - 支援連續特徵（自動尋找最佳分裂閾值）
  - 支援混合型資料（連續 + 離散）
  - 包含後剪枝機制
- **核心類別**：`C45DecisionTree`
- **主要參數**：
  - `max_depth`：最大樹深度
  - `min_samples_split`：內部節點最小樣本數
  - `min_samples_leaf`：葉節點最小樣本數
  - `min_gain_ratio`：最小增益比率閾值
  - `pruning`：是否啟用後剪枝
  - `validation_split`：驗證集比例（用於剪枝）
- **執行方式**：
  ```bash
  cd src
  python hw1_part2_C45_implementation.py
  ```
- **輸出**：
  - 日誌
  - 剪枝前後節點數比較
  - 測試集評估結果
  - 混淆矩陣與分類報告

---

#### `hw1_part2_cart_implementation.py` - CART 決策樹

- **演算法特色**：
  - 使用 Gini 不純度（Gini Impurity）作為分裂標準
  - 生成二元樹（每個節點最多兩個子節點）
  - 支援連續特徵
  - 使用 scikit-learn 實作
- **主要功能**：
  - `train_and_evaluate_cart()`：訓練與評估
  - 計算過擬合差距（Training Acc - Testing Acc）
  - 輸出樹的複雜度統計
- **執行方式**：
  ```bash
  cd src
  python hw1_part2_cart_implementation.py
  ```
- **輸出**：
  - 訓練/測試集準確率
  - 過擬合差距分析
  - 樹節點數與深度
  - 詳細分類報告
  - 混淆矩陣
---

#### `hw1_part2_C50_implementation.py` - C5.0 決策樹

- **演算法特色**：
  - 支援 Boosting（多樹集成）
  - 使用 R 語言的 `C50` 套件實作
- **核心類別**：
  - `C50Experiment`：實驗流程管理器
  - `C50ModelManager`：模型訓練與預測
  - `ModelEvaluator`：評估與結果儲存
  - `AdultDataProcessor`：C5.0 專用資料處理
- **主要參數**：
  - `trials`：Boosting 迭代次數（預設 10）
  - `use_cost_matrix`：是否使用成本矩陣
  - `validation_size`：驗證集比例
  - `n_bins`：離散化區間數
- **執行方式**：
  ```bash
  cd src
  python hw1_part2_C50_implementation.py
  ```
- **輸出**：
  - 日誌（同時輸出到檔案）
  - 訓練/驗證/測試集評估結果
  - 評估報告（`results/evaluation_report.txt`）
  - 預測結果 CSV 檔案
  - 混淆矩陣 CSV 檔案
  - 儲存的 R 模型檔案（`results/c50_model.rds`）

---

### Part 3: CART 剪枝分析

#### `hw1_part3_cart_pruning.py` - 成本複雜度剪枝

- **功能**：分析 CART 決策樹的成本複雜度剪枝（Cost Complexity Pruning）
- **分析內容**：
  1. **剪枝路徑分析**：計算不同 `ccp_alpha` 值下的樹結構
  2. **三種剪枝程度比較**：
     - **輕度剪枝**（Light Pruning, α=0.0）：過擬合，高訓練準確率
     - **最佳剪枝**（Optimal Pruning）：平衡偏差與變異，最高測試準確率
     - **重度剪枝**（Heavy Pruning）：欠擬合，低訓練與測試準確率
  3. **視覺化分析**：
     - Total Impurity vs Effective Alpha
     - Number of Nodes and Depth vs Alpha
     - Accuracy vs Alpha（標註三個關鍵點）
- **執行方式**：
  ```bash
  cd src
  python hw1_part3_cart_pruning.py
  ```
- **輸出檔案**：
  - `total_impurity_vs_alpha.png`：總不純度隨 alpha 變化
  - `nodes_and_depth_vs_alpha.png`：節點數與深度隨 alpha 變化
  - `accuracy_vs_alpha_cart.png`：準確率隨 alpha 變化（含標註點）
  - `cart_tree_ccp_alpha_*.png`：三種剪枝程度的決策樹視覺化
  - Console 輸出詳細的比較表格與分析

---

## 執行方法

### 前置準備

1. **確認資料集位置**：
   ```
   data/
   ├── adult.data
   └── adult.test
   ```

2. **安裝必要套件**：
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

3. **（選用）安裝 C5.0 需要的 R 環境**：
   ```bash
   pip install rpy2
   ```
   然後在 R 中執行：
   ```r
   install.packages("C50")
   ```

---

### 執行順序建議

#### 1. Part 1 - Baseline 模型

```bash
# 使用 Jupyter Notebook
jupyter notebook src/hw1_part1_baseline.ipynb
```

或使用 JupyterLab、VS Code 等工具開啟 notebook 並執行所有 cell。

**預期結果**：
- 訓練準確率：~1.0000（完全過擬合）
- 測試準確率：~0.8047

---

#### 2. Part 2 - 決策樹演算法實作

建議執行順序（從簡單到複雜）：

##### (1) CART 決策樹

```bash
cd src
python hw1_part2_cart_implementation.py
```

---

##### (2) ID3 決策樹

```bash
cd src
python hw1_part2_id3_implementation.py
```

---

##### (3) C4.5 決策樹

```bash
cd src
python hw1_part2_C45_implementation.py
```

---

##### (4) C5.0 決策樹（需要 R）

```bash
cd src
python hw1_part2_C50_implementation.py
```

**預期輸出**：
```
訓練集準確率: ~0.8600
驗證集準確率: ~0.8500
測試集準確率: ~0.8500
```

---

#### 3. Part 3 - CART 剪枝分析

```bash
cd src
python hw1_part3_cart_pruning.py
```

---

## 演算法說明

### 1. ID3 (Iterative Dichotomiser 3)

- **分裂標準**：資訊增益（Information Gain）
- **不純度計算**：熵（Entropy）
- **優點**：
  - 概念簡單，易於理解
  - 適合處理名義型特徵
- **缺點**：
  - 偏好取值較多的特徵
  - 僅支援離散特徵
  - 容易過擬合
- **適用情境**：小型資料集、教學用途

---

### 2. C4.5

- **分裂標準**：增益比率（Gain Ratio）= 資訊增益 / 分裂資訊
- **改進**：
  - 解決 ID3 的偏向問題
  - 支援連續特徵（自動尋找閾值）
  - 包含剪枝機制（減少過擬合）
  - 可處理缺失值
- **優點**：
  - 泛化能力強
  - 支援混合型資料
  - 生成的規則易於解釋
- **缺點**：
  - 訓練時間較長
  - 內存需求較高
- **適用情境**：中等規模資料集、需要模型解釋性的場景

---

### 3. CART (Classification and Regression Trees)

- **分裂標準**：Gini 不純度
- **樹結構**：二元樹（每個節點最多兩個子節點）
- **優點**：
  - 計算效率高
  - 支援回歸任務
  - 生成的樹較平衡
  - scikit-learn 實作成熟
- **缺點**：
  - 無剪枝時容易過擬合
  - 對資料不平衡敏感
- **適用情境**：
  - 大規模資料集
  - 需要快速訓練的場景
  - 回歸與分類任務

---

### 4. C5.0

- **演算法特色**：
  - C4.5 的商業改進版本
  - 支援 Boosting（多樹集成）
  - 訓練速度更快
  - 記憶體效率更高
- **優點**：
  - 高準確率（透過 Boosting）
  - 支援成本敏感學習
  - 可處理大規模資料
- **缺點**：
  - 需要 R 語言環境
  - 商業軟體（開源版本功能有限）
- **適用情境**：
  - 商業應用
  - 需要高準確率的場景
  - 不平衡資料集（使用成本矩陣）

---

## 結果分析

### 預期效能比較

| 演算法 | 測試準確率 | 訓練時間 | 樹大小 | 過擬合程度 |
|--------|-----------|---------|--------|-----------|
| ID3 | ~0.82 | 中 | 中 | 中 |
| C4.5 | ~0.84 | 長 | 中 | 低 |
| CART (無剪枝) | ~0.80 | 短 | 大 | 高 |
| CART (剪枝) | ~0.84 | 中 | 小 | 低 |
| C5.0 | ~0.85 | 中 | 小 | 低 |

---

### Part 3 剪枝分析重點

**偏差-變異權衡（Bias-Variance Tradeoff）**：

1. **輕度剪枝（α=0.0）**：
   - 訓練準確率：~1.0000（過擬合）
   - 測試準確率：~0.8047
   - 特徵：高變異、低偏差

2. **最佳剪枝（α=optimal）**：
   - 訓練準確率：~0.8600
   - 測試準確率：~0.8450（最高）
   - 特徵：平衡偏差與變異

3. **重度剪枝（α=large）**：
   - 訓練準確率：~0.7500
   - 測試準確率：~0.7600
   - 特徵：低變異、高偏差（欠擬合）

---

## 注意事項

### 1. 資料集位置

- 確保 `data/adult.data` 和 `data/adult.test` 存在
- 資料檔案應使用 UTF-8 編碼

### 2. 執行環境

- **Python 版本**：建議 3.8 以上
- **記憶體需求**：至少 4GB RAM
- **磁碟空間**：至少 500MB（包含輸出檔案）

### 3. C5.0 特殊需求

- 需要安裝 R 語言（建議 R 4.0+）
- 需要安裝 `rpy2` 和 R 套件 `C50`
- Windows 用戶需要額外配置 R 路徑

### 4. 執行時間預估

- **Part 1**：< 1 分鐘
- **Part 2 (CART)**：< 2 分鐘
- **Part 2 (ID3)**：5-10 分鐘
- **Part 2 (C4.5)**：10-15 分鐘
- **Part 2 (C5.0)**：5-10 分鐘（需要 R）
- **Part 3**：2-5 分鐘

### 5. 常見問題

**Q1: 找不到資料檔案？**
```
FileNotFoundError: 找不到訓練資料: ../data/adult.data
```
**解決方式**：確認資料集位於正確的 `data/` 目錄下。

**Q2: C5.0 執行失敗？**
```
✗ C50 載入失敗: No module named 'rpy2'
```
**解決方式**：
```bash
pip install rpy2
# 並在 R 中執行
install.packages("C50")
```

---

## 總結

本專案完整實作了四種經典決策樹演算法，並提供：

- ✅ 統一的資料前處理模組
- ✅ 演算法實作
- ✅ 詳細的評估與分析
- ✅ 視覺化結果（Part 3）
- ✅ 可重現的實驗（固定隨機種子）

所有程式均可獨立執行，並提供輸出日誌，方便理解演算法運作原理與效能差異。

---

如有任何問題，請參考程式內的詳細註解或聯繫作者。