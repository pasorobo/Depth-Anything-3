# 02. バッチ画像処理

ディレクトリ内の複数の画像を効率的に処理します。バッチ処理によりGPU利用率を最大化します。

## 🚀 クイックスタート

```bash
python batch_image_processing.py --input ./images --output ./output_batch
```

## 📋 使用方法

### 基本的な使用

```bash
# ディレクトリ内のすべての画像を処理
python batch_image_processing.py --input ./images --output ./output_batch

# バッチサイズを指定
python batch_image_processing.py --input ./images --batch-size 4

# 特定の拡張子のみ処理
python batch_image_processing.py --input ./images --extension jpg,png

# NPZファイルとして保存
python batch_image_processing.py --input ./images --save-npz --save-raw
```

## ⚙️ オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input`, `-i` | 入力画像ディレクトリ | 必須 |
| `--output`, `-o` | 出力ディレクトリ | `./output_batch` |
| `--batch-size`, `-b` | バッチサイズ | `1` |
| `--extension` | 処理する画像拡張子（カンマ区切り） | `jpg,jpeg,png,bmp,tiff` |
| `--save-raw` | 生の深度配列を保存 | False |
| `--save-npz` | バッチ結果をNPZファイルとして保存 | False |
| `--model`, `-m` | モデル名またはパス | `depth-anything/DA3-LARGE` |
| `--process-res` | 処理解像度 | `504` |
| `--device` | 使用デバイス (cuda/cpu) | 自動検出 |

## 📤 出力

- `{filename}_depth.png` - 各画像のカラー化された深度マップ
- `{filename}_depth.npy` - 生の深度データ（`--save-raw` 使用時）
- `batch_results.npz` - すべての結果を含むNPZファイル（`--save-npz` 使用時）

## 💡 使用例

### 例1: 大量の画像を高速処理

```bash
python batch_image_processing.py \
    --input ./dataset \
    --batch-size 8 \
    --output ./results
```

### 例2: JPGのみ処理してNPZで保存

```bash
python batch_image_processing.py \
    --input ./photos \
    --extension jpg \
    --save-npz \
    --output ./depth_results
```

### 例3: 生データも含めて保存

```bash
python batch_image_processing.py \
    --input ./images \
    --save-raw \
    --save-npz \
    --batch-size 4
```

## 🔧 トラブルシューティング

### GPU メモリ不足

```bash
# バッチサイズを減らす
python batch_image_processing.py --input ./images --batch-size 1

# または処理解像度を下げる
python batch_image_processing.py --input ./images --process-res 336
```

### 進捗が遅い

GPUを使用していることを確認してください：
```bash
python batch_image_processing.py --input ./images --device cuda
```

## 📊 パフォーマンス目安

| バッチサイズ | GPU メモリ | 処理速度（目安） |
|------------|-----------|---------------|
| 1 | 4GB | 1.0x |
| 4 | 8GB | 2.5x |
| 8 | 12GB | 4.0x |

## 📚 関連サンプル

- [01. 単一画像推論](../01_single_image_inference/) - 1枚の画像を処理
- [03. 動画処理](../03_video_processing/) - 動画フレームの処理
