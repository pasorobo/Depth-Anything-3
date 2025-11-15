# 06. マルチビュー深度とカメラポーズ推定

同じシーンの複数画像を処理し、深度マップとカメラポーズの両方を推定します。3D再構成やノベルビュー合成に有用です。

## 🚀 クイックスタート

```bash
python multiview_pose_estimation.py --input images/ --output ./output_multiview
```

## 📋 使用方法

### 基本的な使用

```bash
# マルチビュー処理
python multiview_pose_estimation.py --input images/ --output ./output_multiview

# カメラポーズの3D可視化
python multiview_pose_estimation.py --input images/ --visualize-cameras

# GLBファイルとしてエクスポート
python multiview_pose_estimation.py --input images/ --export-glb

# 処理する画像数を制限
python multiview_pose_estimation.py --input images/ --max-images 10
```

## ⚙️ オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input`, `-i` | 画像ディレクトリ | 必須 |
| `--output`, `-o` | 出力ディレクトリ | `./output_multiview` |
| `--model`, `-m` | モデル名またはパス | `depth-anything/DA3-LARGE` |
| `--max-images` | 処理する最大画像数 | なし |
| `--process-res` | 処理解像度 | `504` |
| `--device` | 使用デバイス (cuda/cpu) | 自動検出 |
| `--visualize-cameras` | カメラポーズの3D可視化 | False |
| `--export-glb` | 3D再構成をGLBとしてエクスポート | False |

## 📤 出力

- `{filename}_depth.png` - 各画像の深度マップ
- `camera_poses.png` - カメラポーズの3D可視化（`--visualize-cameras`使用時）
- `multiview_data.npz` - すべてのデータ（深度、ポーズ、内部パラメータ）
- `scene.glb` - 3D再構成（`--export-glb` 使用時）

## 💡 使用例

### 例1: カメラポーズの可視化

```bash
python multiview_pose_estimation.py \
    --input ./scene_images/ \
    --visualize-cameras \
    --output ./pose_analysis
```

### 例2: 完全な3D再構成

```bash
python multiview_pose_estimation.py \
    --input ./photos/ \
    --export-glb \
    --visualize-cameras
```

### 例3: 一部の画像のみ処理

```bash
python multiview_pose_estimation.py \
    --input ./large_dataset/ \
    --max-images 20 \
    --visualize-cameras
```

## 📊 カメラポーズ情報

### Extrinsics（外部パラメータ）
- **World-to-Camera 変換行列** (4×4 または 3×4)
- カメラの位置と姿勢を表現
- 3D空間での各カメラの配置

### Intrinsics（内部パラメータ）
- **焦点距離**: fx, fy
- **主点**: cx, cy
- カメラの光学特性

### 可視化例

`camera_poses.png` には以下が表示されます：
- 🔴 カメラ位置（赤い点）
- 🔵 視線方向（青い矢印）
- 📝 カメララベル

## 🎯 実用的な使用シーン

### 3Dスキャン
```bash
# オブジェクトの周りから撮影した画像
python multiview_pose_estimation.py \
    --input ./object_360/ \
    --export-glb \
    --visualize-cameras
```

### 建築記録
```bash
# 建物の異なる角度からの写真
python multiview_pose_estimation.py \
    --input ./building_photos/ \
    --visualize-cameras
```

### SfM (Structure from Motion)
```bash
# カメラの軌跡を推定
python multiview_pose_estimation.py \
    --input ./sequence/ \
    --visualize-cameras \
    --max-images 50
```

## 🔧 トラブルシューティング

### カメラポーズが推定されない

`DA3-LARGE`以上のモデルを使用していることを確認：
```bash
python multiview_pose_estimation.py --input images/ --model depth-anything/DA3-LARGE
```

### 画像が多すぎて処理が遅い

`--max-images`で画像数を制限：
```bash
python multiview_pose_estimation.py --input images/ --max-images 30
```

## 📐 撮影のコツ

### 良い結果を得るために

1. **重複する視野**: 隣接画像で50%以上重複
2. **十分な視差**: カメラを適度に移動
3. **一定の照明**: 撮影中の光源変化を最小限に
4. **テクスチャ豊富**: 平坦な壁より特徴的な表面

### 推奨撮影パターン

- **円形**: オブジェクトの周りを一周
- **グリッド**: 建物の正面を格子状に
- **螺旋**: 高さを変えながら回転

## 📚 関連サンプル

- [04. 3D再構成](../04_3d_reconstruction/) - 基本的な3D再構成
- [07. 大規模3D再構成](../07_advanced_3d_reconstruction/) - 動画からの大規模再構成
