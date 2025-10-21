# 決策樹演算法實作與分析

**課程作業：資料探勘-決策樹分類器專題**

---

## 📋 目錄

1. [專案概述](#專案概述)
2. [專案架構](#專案架構)
3. [環境需求](#環境需求)
4. [資料集說明](#資料集說明)
5. [程式檔案說明](#程式檔案說明)
6. [演算法說明](#演算法說明)
7. [結果分析](#結果分析)
8. [參考資料](#參考資料)

---

## 專案概述

專案包含三個主要部分：

- **Part 1**：使用 scikit-learn 建立 baseline 模型
- **Part 2**：手動實作 ID3、C4.5、CART、C5.0 四種決策樹演算法
- **Part 3**：CART 決策樹的成本複雜度剪枝分析

---

## 專案架構

```
hw1_m11435021/
│
├── data/                               # 資料集目錄
│   ├── adult.data                      # 訓練資料
│   └── adult.test                      # 測試資料
│
├── src/                                 # 原始碼目錄
│   ├── hw1_part1_baseline.ipynb         # Part 1: Baseline 模型(.ipynb)
│   ├── hw1_part2_preprocessing.py       # Part 2: 統一資料前處理模組
│   ├── hw1_part2_id3_implementation.py  # Part 2: ID3 決策樹實作
│   ├── hw1_part2_C45_implementation.py  # Part 2: C4.5 決策樹實作
│   ├── hw1_part2_cart_implementation.py # Part 2: CART 決策樹實作
│   ├── hw1_part2_C50_implementation.py  # Part 2: C5.0 決策樹實作
│   ├── hw1_part3_cart_pruning.py        # Part 3: CART 剪枝分析
│   ├── create_comparison_excel.py       # Part 2: 將 ID3、C4.5、C5.0、CART 結果輸出 Excel
│   ├── hw1_part2_C45_implementation.ipynb  # Part 2: C4.5 決策樹實作(.ipynb)
│   ├── hw1_part2_C50_implementation.ipynb  # Part 2: C5.0 決策樹實作(.ipynb)
│   ├── hw1_part2_cart_implementation.ipynb  # Part 2: cart 決策樹實作(.ipynb)
│   ├── hw1_part2_id3_implementation.ipynb  # Part 2: ID3 決策樹實作(.ipynb)
│   ├── hw1_part2_preprocessing.ipynb    # Part 2: 統一資料前處理模組(.ipynb)
│   └── hw1_part3_cart_pruning.ipynb     # Part 3: CART 剪枝分析(.ipynb)
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

- **功能**：使用 scikit-learn 訓練 CART 模型，包括資料清理、One-Hot 編碼、準確率計算。
- **執行方式**：使用 Jupyter Notebook 開啟並執行（已有跑過的過程）

---

### Part 2: 決策樹演算法實作

#### `hw1_part2_preprocessing.py` - 統一資料前處理模組

- **功能**：提供 Part2 演算法統一的資料前處理介面，支援缺失值處理、編碼、離散化、驗證集分割
- **核心類別**：`UnifiedDataPreprocessor`
- **參數調整**：
  ```python
  discretize=False,  # 是否離散化
  ```
- **執行方式**：使用 Jupyter Notebook 開啟並執行（已有跑過的過程）

---

#### `hw1_part2_id3_implementation.py` - ID3 決策樹

- **功能**：ID3（資訊增益、熵、後剪枝）
- **核心類別**：`ID3DecisionTree`
- **執行方式**：使用 Jupyter Notebook 開啟並執行（已有跑過的過程）
- **輸出**：
  - 訓練/驗證/測試集準確率
  - 混淆矩陣與分類報告
  - 決策樹統計資訊（節點數、深度）
  - 儲存模型檔案 `id3_model.pkl`

---

#### `hw1_part2_C45_implementation.py` - C4.5 決策樹

- **功能**：C4.5（增益比率、支援連續特徵、後剪枝）
- **核心類別**：`C45DecisionTree`
- **執行方式**：使用 Jupyter Notebook 開啟並執行（已有跑過的過程）
- **輸出**：
  - 日誌
  - 剪枝前後節點數比較
  - 測試集評估結果
  - 混淆矩陣與分類報告

---

#### `hw1_part2_cart_implementation.py` - CART 決策樹

- **功能**：CART（Gini 不純度、二元樹）
- **核心類別**：`train_and_evaluate_cart()`
- **執行方式**：使用 Jupyter Notebook 開啟並執行（已有跑過的過程）
- **輸出**：
  - 訓練/測試集準確率
  - 過擬合差距分析
  - 樹節點數與深度
  - 詳細分類報告
  - 混淆矩陣
---

#### `hw1_part2_C50_implementation.py` - C5.0 決策樹

- **功能**：C5.0（Boosting，使用 R C50）
- **核心類別**：
  - `C50Experiment`：實驗流程管理器
  - `C50ModelManager`：模型訓練與預測
  - `ModelEvaluator`：評估與結果儲存
  - `AdultDataProcessor`：C5.0 專用資料處理
- **執行方式**：使用 Jupyter Notebook 開啟並執行（已有跑過的過程）
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

- **功能**：分析 ccp_alpha 下的剪枝路徑、輕/最佳/重度剪枝比較、視覺化（不純度、節點/深度、準確率）
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
- **執行方式**：使用 Jupyter Notebook 開啟並執行（已有跑過的過程）
- **[輸出檔案](https://github.com/ovi-saries/hw1_m11435021/tree/main/results/part3)**：
  - `total_impurity_vs_alpha.png`：總不純度隨 alpha 變化
  - `nodes_and_depth_vs_alpha.png`：節點數與深度隨 alpha 變化
  - `accuracy_vs_alpha_cart.png`：準確率隨 alpha 變化（含標註點）
  - `cart_tree_ccp_alpha_*.png`：三種剪枝程度的決策樹視覺化

---

## 演算法說明

- ID3：**資訊增益分裂，僅離散特徵**。適用情境：小型資料集、教學用途
- C4.5：**增益比率**，支援連續/離散，後剪枝**。適用情境：中等規模資料集、需要模型解釋性的場景
- CART：**Gini 不純度，二元樹**。適用情境：大規模資料集、需要快速訓練的場景、回歸與分類任務
- C5.0：**Boosting 集成**。適用情境：商業應用、需要高準確率的場景、不平衡資料集（使用成本矩陣）

> Boosting 先用一棵決策樹預測資料，然後讓後續樹專注修正前一棵的錯誤，重複幾次後合併預測，提升整體準確率。

---

## 結果分析

### 預期效能比較

| 演算法 | 訓練準確度 | 測試準確率 | 
|--------|-----------|---------|
| ID3 (no prune) | 0.8719 | 0.8079 |
| ID3 (prune) | 0.8452 | 0.8231 |
| C4.5 | 0.8439 | 0.8398 |
| CART (no prune) | 1.0000 | 0.8024 |
| CART (optimal) | 0.8702 | 0.8526 |
| C5.0 | 0.9580 | 0.8491 |

測試準確率由高到低
CART (optimal) (0.8526) > C5.0 (0.8491) > C4.5 (0.8398) > ID3 (prune) (0.8231) > ID3 (no prune) (0.8079) > CART (no prune) (0.8024)

[Part2_Excel](https://github.com/ovi-saries/hw1_m11435021/blob/main/results/hw1_part2_decision_trees_comparison.xlsx)

---

## 參考資料

[1]. Becker, B., & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/2/adult

[2]. Esmer, B. (n.d.). C4.5: A Python implementation of C4.5 algorithm by R. Quinlan. GitHub. https://github.com/barisesmer/C4.5?tab=readme-ov-file

[3]. GeeksforGeeks. (2025, July 23). Sklearn | Iterative Dichotomiser 3 (ID3) Algorithms. https://www.geeksforgeeks.org/machine-learning/sklearn-iterative-dichotomiser-3-id3-algorithms/

[4]. R Core Team. (n.d.). C50: C5.0 Decision Trees and Rule-Based Models. CRAN. https://cran.r-project.org/package=C50

[5]. Salzberg, S. L. (1994). Book Review: C4.5: Programs for Machine Learning by J. Ross Quinlan. Machine Learning, 16(3), 235-240. https://link.springer.com/article/10.1023/A:1022645310020

[6]. Scikit-learn developers. (n.d.). Cost Complexity Pruning. https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py

[7]. Scikit-learn developers. (n.d.). DecisionTreeClassifier. Scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html