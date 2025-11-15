# 01. 基本的な単一画像推論

最もシンプルな使用例。単一の画像から深度マップを推定し、可視化結果を保存します。

## 🚀 クイックスタート

```bash
python single_image_inference.py --input image.jpg --output ./output
```

## 📋 使用方法

### 基本的な使用

```bash
# 基本的な使用
python single_image_inference.py --input path/to/image.jpg --output ./output

# カスタムモデルを使用
python single_image_inference.py --input image.jpg --model depth-anything/DA3-BASE

# CPU で実行
python single_image_inference.py --input image.jpg --device cpu

# 異なるカラーマップを使用
python single_image_inference.py --input image.jpg --colormap viridis
```

## ⚙️ オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input`, `-i` | 入力画像のパス | 必須 |
| `--output`, `-o` | 出力ディレクトリ | `./output` |
| `--model`, `-m` | モデル名またはパス | `depth-anything/DA3-LARGE` |
| `--process-res` | 処理解像度 | `504` |
| `--device` | 使用デバイス (cuda/cpu) | 自動検出 |
| `--colormap` | 可視化用カラーマップ | `Spectral` |

## 📤 出力

- `{filename}_depth.npy` - 生の深度マップ（NumPy配列）
- `{filename}_depth_vis.png` - カラー化された深度マップ
- `{filename}_confidence.npy` - 信頼度マップ（利用可能な場合）
- `{filename}_confidence_vis.png` - カラー化された信頼度マップ
- `{filename}_comparison.png` - 元画像と深度マップの並列比較

## 💡 使用例

### 例1: シンプルな深度推定

```bash
python single_image_inference.py --input photo.jpg
```

### 例2: 高解像度処理

```bash
python single_image_inference.py --input photo.jpg --process-res 672
```

### 例3: 軽量モデルで高速処理

```bash
python single_image_inference.py --input photo.jpg --model depth-anything/DA3-SMALL
```

## 🔧 トラブルシューティング

### GPU メモリ不足

```bash
# 解像度を下げる
python single_image_inference.py --input photo.jpg --process-res 336

# または CPU を使用
python single_image_inference.py --input photo.jpg --device cpu
```

### モデルのダウンロード

初回実行時、モデルは自動的に Hugging Face Hub からダウンロードされます。インターネット接続が必要です。

## 📚 関連サンプル

- [02. バッチ画像処理](../02_batch_image_processing/) - 複数画像の一括処理
- [05. メトリック深度推定](../05_metric_depth/) - 実スケールの深度推定
