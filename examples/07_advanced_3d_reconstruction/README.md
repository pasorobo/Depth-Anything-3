# 07. 大規模3D再構成（動画・画像シーケンス）

動画ファイルや画像シーケンスから大規模な3D再構成を実行します。すべてのフレームを処理し、深度とカメラポーズを推定して、統合された点群を生成します。

## 🚀 クイックスタート

```bash
python advanced_3d_reconstruction.py --input video.mp4 --output ./reconstruction
```

## 📋 使用方法

### 基本的な使用

```bash
# 動画から3D再構成
python advanced_3d_reconstruction.py --input video.mp4 --output ./reconstruction

# 画像ディレクトリから3D再構成
python advanced_3d_reconstruction.py --input images/ --output ./reconstruction

# FPSを指定してフレームを間引く
python advanced_3d_reconstruction.py --input video.mp4 --fps 5 --max-frames 100

# 高品質再構成（大きいモデル、高解像度）
python advanced_3d_reconstruction.py --input video.mp4 \
    --model depth-anything/DA3-GIANT --process-res 672

# バッチ処理で高速化
python advanced_3d_reconstruction.py --input video.mp4 --batch-size 8

# 点群のダウンサンプリングと信頼度フィルタリング
python advanced_3d_reconstruction.py --input video.mp4 \
    --downsample-points 0.5 --min-confidence 0.7
```

## ⚙️ オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input`, `-i` | 動画ファイルまたは画像ディレクトリ | 必須 |
| `--output`, `-o` | 出力ディレクトリ | `./reconstruction` |
| `--model`, `-m` | モデル名またはパス | `depth-anything/DA3-LARGE` |
| `--fps` | 動画処理のターゲットFPS | すべてのフレーム |
| `--max-frames` | 処理する最大フレーム数 | なし |
| `--process-res` | 処理解像度 | `504` |
| `--device` | 使用デバイス (cuda/cpu) | 自動検出 |
| `--batch-size` | バッチサイズ | `8` |
| `--downsample-points` | 点群ダウンサンプリング率（0-1） | `1.0` (ダウンサンプリングなし) |
| `--min-confidence` | 点の最小信頼度閾値 | `0.5` |
| `--export-format` | エクスポート形式（カンマ区切り） | `glb,ply,npz` |

## 📤 出力

- `reconstruction.ply` - PLY形式の点群
- `reconstruction.npz` - NPZ形式のデータ（点座標、色、メタデータ）
- `reconstruction.glb` - GLB形式の3Dモデル
- `reconstruction_info.txt` - 再構成情報（フレーム数、点数など）

## 💡 使用例

### 例1: 動画から高品質3Dモデル

```bash
python advanced_3d_reconstruction.py \
    --input scan_video.mp4 \
    --fps 10 \
    --model depth-anything/DA3-GIANT \
    --process-res 672 \
    --output ./high_quality
```

### 例2: 画像シーケンスから軽量モデル

```bash
python advanced_3d_reconstruction.py \
    --input ./image_sequence/ \
    --downsample-points 0.3 \
    --min-confidence 0.7 \
    --export-format ply \
    --output ./light_model
```

### 例3: 大規模シーンの高速処理

```bash
python advanced_3d_reconstruction.py \
    --input large_scene.mp4 \
    --fps 5 \
    --batch-size 16 \
    --downsample-points 0.5 \
    --output ./large_scene_recon
```

## ⚡ パフォーマンス最適化

### GPU メモリに応じたバッチサイズ

| GPU VRAM | 推奨バッチサイズ | 解像度 |
|----------|----------------|--------|
| 4GB | 2 | 336 |
| 8GB | 4-8 | 504 |
| 12GB+ | 8-16 | 504-672 |

### 点群サイズの削減

```bash
# フィルタリングで点数を減らす
--min-confidence 0.7  # 信頼度の高い点のみ
--downsample-points 0.5  # 50%にダウンサンプリング

# 処理後にさらに削減
cd ../utils
python pointcloud_tools.py downsample \
    --input ../07_advanced_3d_reconstruction/reconstruction.ply \
    --output ../07_advanced_3d_reconstruction/reconstruction_small.ply \
    --voxel-size 0.05
```

## 🎬 撮影のコツ

### 動画撮影時の推奨事項

1. **スムーズな動き**: カメラを滑らかに動かす
2. **適切な速度**: ゆっくりと移動（1m/秒以下）
3. **重複確保**: 各フレームで70-80%の重複
4. **照明一定**: 撮影中の光源変化を避ける
5. **手ブレ対策**: 可能なら三脚やジンバル使用

### 理想的なカメラ軌跡

- **円形軌道**: オブジェクト中心に円を描く
- **螺旋軌道**: 高さを変えながら回転
- **直線軌道**: 建物の前を平行移動

## 🔧 トラブルシューティング

### 点群が巨大すぎる

```bash
# ダウンサンプリングと信頼度フィルタリングを使用
python advanced_3d_reconstruction.py \
    --input video.mp4 \
    --downsample-points 0.3 \
    --min-confidence 0.7
```

### メモリ不足エラー

```bash
# バッチサイズと解像度を下げる
python advanced_3d_reconstruction.py \
    --input video.mp4 \
    --batch-size 2 \
    --process-res 336
```

### GLBエクスポートに失敗

```bash
# PLYとNPZのみでエクスポート
python advanced_3d_reconstruction.py \
    --input video.mp4 \
    --export-format ply,npz
```

## 📊 処理時間の目安

1分間の1080p動画（30fps）の場合:

| 設定 | フレーム数 | 処理時間（RTX 3080） |
|------|----------|-------------------|
| FPS=30, Batch=1 | 1800 | ~60分 |
| FPS=10, Batch=8 | 600 | ~10分 |
| FPS=5, Batch=8 | 300 | ~5分 |

## 📚 関連サンプル

- [03. 動画処理](../03_video_processing/) - 深度動画の作成
- [06. マルチビュー深度とカメラポーズ推定](../06_multiview_pose_estimation/) - 基本的なマルチビュー処理
- [08. ライブカメラ3D再構成](../08_live_camera_reconstruction/) - リアルタイム再構成
- [ユーティリティツール](../utils/) - 点群の後処理
