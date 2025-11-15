# 03. 動画処理

動画ファイルからフレームを抽出し、各フレームの深度を推定します。結果は個別フレームまたは動画として保存できます。

## 🚀 クイックスタート

```bash
python video_processing.py --input video.mp4 --output ./output_video
```

## 📋 使用方法

### 基本的な使用

```bash
# 動画を処理
python video_processing.py --input video.mp4 --output ./output_video

# FPSを指定してフレームを間引く
python video_processing.py --input video.mp4 --fps 10

# 深度動画を作成
python video_processing.py --input video.mp4 --create-video

# 元動画と深度マップの並列比較動画を作成
python video_processing.py --input video.mp4 --side-by-side --save-frames
```

## ⚙️ オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input`, `-i` | 入力動画ファイル | 必須 |
| `--output`, `-o` | 出力ディレクトリ | `./output_video` |
| `--model`, `-m` | モデル名またはパス | `depth-anything/DA3-LARGE` |
| `--fps` | 処理するFPS（フレーム間引き用） | すべてのフレーム |
| `--max-frames` | 処理する最大フレーム数 | なし |
| `--process-res` | 処理解像度 | `504` |
| `--device` | 使用デバイス (cuda/cpu) | 自動検出 |
| `--create-video` | 深度マップから動画を作成 | False |
| `--side-by-side` | 並列比較動画を作成 | False |
| `--save-frames` | 個別フレームを画像として保存 | False |

## 📤 出力

- `frames/{frame_number}_depth.png` - 個別の深度フレーム（`--save-frames` 使用時）
- `depth_output.mp4` - 深度マップ動画（`--create-video` 使用時）
- `comparison.mp4` - 並列比較動画（`--side-by-side` 使用時）
- `depth_data.npz` - すべてのフレームの深度データ

## 💡 使用例

### 例1: 深度動画を作成

```bash
python video_processing.py \
    --input walking_tour.mp4 \
    --create-video \
    --fps 10
```

### 例2: 元動画と深度の比較動画

```bash
python video_processing.py \
    --input scene.mp4 \
    --side-by-side \
    --fps 15
```

### 例3: 個別フレームとして保存

```bash
python video_processing.py \
    --input video.mp4 \
    --save-frames \
    --max-frames 100
```

## 🔧 トラブルシューティング

### 処理が遅い

FPSを下げてフレーム数を減らしてください：
```bash
python video_processing.py --input video.mp4 --fps 5
```

### メモリ不足

最大フレーム数を制限してください：
```bash
python video_processing.py --input video.mp4 --max-frames 50
```

## 📊 処理時間目安

30秒の1080p動画（30fps）の場合:

| FPS設定 | フレーム数 | 処理時間（GPU） |
|---------|----------|--------------|
| 30 (全て) | 900 | ~15分 |
| 10 | 300 | ~5分 |
| 5 | 150 | ~2.5分 |
| 1 | 30 | ~30秒 |

## 📚 関連サンプル

- [02. バッチ画像処理](../02_batch_image_processing/) - 画像の一括処理
- [07. 大規模3D再構成](../07_advanced_3d_reconstruction/) - 動画から3D再構成
