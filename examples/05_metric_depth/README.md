# 05. メトリック深度推定

実際のスケール（メートル単位）で深度を推定します。相対深度ではなく、実際の距離測定を提供します。

## 🚀 クイックスタート

```bash
python metric_depth.py --input image.jpg --output ./output_metric
```

## 📋 使用方法

### 基本的な使用

```bash
# メトリック深度推定
python metric_depth.py --input image.jpg --output ./output_metric

# 深度範囲の可視化を含める
python metric_depth.py --input image.jpg --visualize-ranges

# 異なる単位で表示
python metric_depth.py --input image.jpg --depth-unit centimeters
python metric_depth.py --input image.jpg --depth-unit feet
```

## ⚙️ オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input`, `-i` | 入力画像 | 必須 |
| `--output`, `-o` | 出力ディレクトリ | `./output_metric` |
| `--model`, `-m` | モデル名 | `depth-anything/DA3METRIC-LARGE` |
| `--process-res` | 処理解像度 | `504` |
| `--device` | 使用デバイス (cuda/cpu) | 自動検出 |
| `--visualize-ranges` | 深度範囲別の可視化を作成 | False |
| `--depth-unit` | 深度表示単位（meters, centimeters, feet） | `meters` |
| `--colormap` | 可視化用カラーマップ | `Spectral` |

## 📤 出力

- `{filename}_depth_metric.npy` - メトリック深度データ（メートル単位）
- `{filename}_depth_vis.png` - カラー化された深度マップ
- `{filename}_histogram.png` - 深度分布ヒストグラム
- `{filename}_depth_ranges.png` - 深度範囲別の可視化（`--visualize-ranges`使用時）
- `{filename}_stats.txt` - 統計情報テキストファイル

## 💡 使用例

### 例1: 完全な分析レポート

```bash
python metric_depth.py \
    --input room.jpg \
    --visualize-ranges \
    --output ./analysis
```

### 例2: センチメートル単位で測定

```bash
python metric_depth.py \
    --input object.jpg \
    --depth-unit centimeters
```

### 例3: フィート単位（建築用）

```bash
python metric_depth.py \
    --input building.jpg \
    --depth-unit feet \
    --visualize-ranges
```

## 📊 深度統計の読み方

生成される統計ファイルには以下の情報が含まれます：

```
Metric Depth Statistics
========================================
Image: room.jpg
Model: depth-anything/DA3METRIC-LARGE
Units: m

Mean depth:   3.45 m    # 平均距離
Median depth: 3.21 m    # 中央値
Min depth:    0.52 m    # 最も近い点
Max depth:    8.93 m    # 最も遠い点
Std dev:      1.23 m    # 標準偏差
```

## 🎯 実用的な使用シーン

### 室内測定
```bash
python metric_depth.py --input room.jpg --visualize-ranges
```
- 部屋の寸法推定
- 家具配置の計画
- リノベーション計画

### 建築現場
```bash
python metric_depth.py --input construction.jpg --depth-unit feet
```
- 距離測定
- 高さ確認
- 進捗記録

### 屋外シーン
```bash
python metric_depth.py --input outdoor.jpg --visualize-ranges
```
- 障害物までの距離
- 地形分析

## 🔧 トラブルシューティング

### 深度値が不正確

メトリックモデルを使用していることを確認：
```bash
python metric_depth.py --input image.jpg --model depth-anything/DA3METRIC-LARGE
```

### 範囲可視化が生成されない

`--visualize-ranges` フラグを追加：
```bash
python metric_depth.py --input image.jpg --visualize-ranges
```

## 📚 関連サンプル

- [01. 単一画像推論](../01_single_image_inference/) - 相対深度推定
- [06. マルチビュー深度とカメラポーズ推定](../06_multiview_pose_estimation/) - 複数視点からの測定
