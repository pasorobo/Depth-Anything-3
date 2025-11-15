# Depth Anything 3 - Examples

このディレクトリには、Depth Anything 3を使用した実用的なサンプルコードが含まれています。各サンプルは、WindowsとLinuxの両方のコマンドラインで動作するように設計されています。

## 📋 目次

- [セットアップ](#セットアップ)
- [サンプル一覧](#サンプル一覧)
  - [01. 基本的な単一画像推論](#01-基本的な単一画像推論)
  - [02. バッチ画像処理](#02-バッチ画像処理)
  - [03. 動画処理](#03-動画処理)
  - [04. 3D再構成](#04-3d再構成)
  - [05. メトリック深度推定](#05-メトリック深度推定)
  - [06. マルチビュー深度とカメラポーズ推定](#06-マルチビュー深度とカメラポーズ推定)
- [よくある質問](#よくある質問)

## 🚀 セットアップ

### 1. 基本インストール

```bash
# リポジトリのルートディレクトリから実行
pip install -e .

# または、すべての機能を含めてインストール
pip install -e ".[all]"
```

### 2. サンプル用の追加パッケージ

```bash
# examples ディレクトリに移動
cd examples

# 追加パッケージのインストール
pip install matplotlib tqdm opencv-python
```

### 3. モデルのダウンロード

サンプルを初めて実行すると、モデルが自動的にHugging Face Hubからダウンロードされます。

## 📚 サンプル一覧

### 01. 基本的な単一画像推論

**ファイル**: `01_single_image_inference.py`

最もシンプルな使用例。単一の画像から深度マップを推定し、可視化結果を保存します。

#### 使用方法

```bash
# 基本的な使用
python 01_single_image_inference.py --input path/to/image.jpg --output ./output

# カスタムモデルを使用
python 01_single_image_inference.py --input image.jpg --model depth-anything/DA3-BASE

# CPU で実行
python 01_single_image_inference.py --input image.jpg --device cpu
```

#### オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input`, `-i` | 入力画像のパス | 必須 |
| `--output`, `-o` | 出力ディレクトリ | `./output` |
| `--model`, `-m` | モデル名またはパス | `depth-anything/DA3-LARGE` |
| `--process-res` | 処理解像度 | `504` |
| `--device` | 使用デバイス (cuda/cpu) | 自動検出 |
| `--colormap` | 可視化用カラーマップ | `Spectral` |

#### 出力

- `{filename}_depth.npy` - 生の深度マップ（NumPy配列）
- `{filename}_depth_vis.png` - カラー化された深度マップ
- `{filename}_confidence.npy` - 信頼度マップ（利用可能な場合）
- `{filename}_confidence_vis.png` - カラー化された信頼度マップ
- `{filename}_comparison.png` - 元画像と深度マップの並列比較

---

### 02. バッチ画像処理

**ファイル**: `02_batch_image_processing.py`

ディレクトリ内の複数の画像を効率的に処理します。バッチ処理によりGPU利用率を最大化します。

#### 使用方法

```bash
# ディレクトリ内のすべての画像を処理
python 02_batch_image_processing.py --input ./images --output ./output_batch

# バッチサイズを指定
python 02_batch_image_processing.py --input ./images --batch-size 4

# 特定の拡張子のみ処理
python 02_batch_image_processing.py --input ./images --extension jpg,png

# NPZファイルとして保存
python 02_batch_image_processing.py --input ./images --save-npz --save-raw
```

#### オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input`, `-i` | 入力画像ディレクトリ | 必須 |
| `--output`, `-o` | 出力ディレクトリ | `./output_batch` |
| `--batch-size`, `-b` | バッチサイズ | `1` |
| `--extension` | 処理する画像拡張子（カンマ区切り） | `jpg,jpeg,png,bmp,tiff` |
| `--save-raw` | 生の深度配列を保存 | False |
| `--save-npz` | バッチ結果をNPZファイルとして保存 | False |

#### 出力

- `{filename}_depth.png` - 各画像のカラー化された深度マップ
- `{filename}_depth.npy` - 生の深度データ（`--save-raw` 使用時）
- `batch_results.npz` - すべての結果を含むNPZファイル（`--save-npz` 使用時）

---

### 03. 動画処理

**ファイル**: `03_video_processing.py`

動画ファイルからフレームを抽出し、各フレームの深度を推定します。結果は個別フレームまたは動画として保存できます。

#### 使用方法

```bash
# 動画を処理
python 03_video_processing.py --input video.mp4 --output ./output_video

# FPSを指定してフレームを間引く
python 03_video_processing.py --input video.mp4 --fps 10

# 深度動画を作成
python 03_video_processing.py --input video.mp4 --create-video

# 元動画と深度マップの並列比較動画を作成
python 03_video_processing.py --input video.mp4 --side-by-side --save-frames
```

#### オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input`, `-i` | 入力動画ファイル | 必須 |
| `--fps` | 処理するFPS（フレーム間引き用） | すべてのフレーム |
| `--max-frames` | 処理する最大フレーム数 | なし |
| `--create-video` | 深度マップから動画を作成 | False |
| `--side-by-side` | 並列比較動画を作成 | False |
| `--save-frames` | 個別フレームを画像として保存 | False |

#### 出力

- `frames/{frame_number}_depth.png` - 個別の深度フレーム（`--save-frames` 使用時）
- `depth_output.mp4` - 深度マップ動画（`--create-video` 使用時）
- `comparison.mp4` - 並列比較動画（`--side-by-side` 使用時）
- `depth_data.npz` - すべてのフレームの深度データ

---

### 04. 3D再構成

**ファイル**: `04_3d_reconstruction.py`

深度マップから3Dポイントクラウドやメッシュを生成します。GLB、PLY、NPZ形式でエクスポート可能です。

#### 使用方法

```bash
# 単一画像から3D再構成
python 04_3d_reconstruction.py --input image.jpg --output ./output_3d

# 複数画像から3D再構成
python 04_3d_reconstruction.py --input images/ --output ./output_3d

# 複数形式でエクスポート
python 04_3d_reconstruction.py --input image.jpg --format glb-npz

# すべての形式でエクスポート
python 04_3d_reconstruction.py --input image.jpg --export-all
```

#### オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input`, `-i` | 入力画像またはディレクトリ | 必須 |
| `--format` | エクスポート形式（glb, npz, mini_npz）<br>`-`で複数指定可能 | `glb` |
| `--export-all` | すべての形式でエクスポート | False |
| `--point-size` | 3D可視化用のポイントサイズ | `0.01` |

#### 出力

- `{name}.glb` - 3Dメッシュ（GLB形式）
- `{name}.npz` - 完全なデータセット
- `{name}_mini.npz` - コンパクト版データセット

#### 3Dファイルの表示方法

- **オンライン**: https://3dviewer.net/
- **Blender**: File > Import > glTF 2.0
- **Windows**: 3D Viewer アプリ
- **その他**: MeshLab, CloudCompare など

---

### 05. メトリック深度推定

**ファイル**: `05_metric_depth.py`

実際のスケール（メートル単位）で深度を推定します。相対深度ではなく、実際の距離測定を提供します。

#### 使用方法

```bash
# メトリック深度推定
python 05_metric_depth.py --input image.jpg --output ./output_metric

# 深度範囲の可視化を含める
python 05_metric_depth.py --input image.jpg --visualize-ranges

# 異なる単位で表示
python 05_metric_depth.py --input image.jpg --depth-unit centimeters
python 05_metric_depth.py --input image.jpg --depth-unit feet
```

#### オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input`, `-i` | 入力画像 | 必須 |
| `--model`, `-m` | モデル名 | `depth-anything/DA3METRIC-LARGE` |
| `--visualize-ranges` | 深度範囲別の可視化を作成 | False |
| `--depth-unit` | 深度表示単位（meters, centimeters, feet） | `meters` |

#### 出力

- `{filename}_depth_metric.npy` - メトリック深度データ（メートル単位）
- `{filename}_depth_vis.png` - カラー化された深度マップ
- `{filename}_histogram.png` - 深度分布ヒストグラム
- `{filename}_depth_ranges.png` - 深度範囲別の可視化
- `{filename}_stats.txt` - 統計情報テキストファイル

---

### 06. マルチビュー深度とカメラポーズ推定

**ファイル**: `06_multiview_pose_estimation.py`

同じシーンの複数画像を処理し、深度マップとカメラポーズの両方を推定します。3D再構成やノベルビュー合成に有用です。

#### 使用方法

```bash
# マルチビュー処理
python 06_multiview_pose_estimation.py --input images/ --output ./output_multiview

# カメラポーズの3D可視化
python 06_multiview_pose_estimation.py --input images/ --visualize-cameras

# GLBファイルとしてエクスポート
python 06_multiview_pose_estimation.py --input images/ --export-glb

# 処理する画像数を制限
python 06_multiview_pose_estimation.py --input images/ --max-images 10
```

#### オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input`, `-i` | 画像ディレクトリ | 必須 |
| `--max-images` | 処理する最大画像数 | なし |
| `--visualize-cameras` | カメラポーズの3D可視化 | False |
| `--export-glb` | 3D再構成をGLBとしてエクスポート | False |

#### 出力

- `{filename}_depth.png` - 各画像の深度マップ
- `camera_poses.png` - カメラポーズの3D可視化
- `multiview_data.npz` - すべてのデータ（深度、ポーズ、内部パラメータ）
- `scene.glb` - 3D再構成（`--export-glb` 使用時）

---

## 🔧 よくある質問

### Q: GPU メモリ不足エラーが発生します

A: 以下の方法を試してください：

1. バッチサイズを減らす: `--batch-size 1`
2. 処理解像度を下げる: `--process-res 336`
3. より小さいモデルを使用: `--model depth-anything/DA3-BASE`

### Q: どのモデルを使用すべきですか？

A: 用途に応じて選択してください：

- **一般的な深度推定**: `depth-anything/DA3-LARGE`（推奨）
- **高速処理**: `depth-anything/DA3-BASE` または `DA3-SMALL`
- **メトリック深度**: `depth-anything/DA3METRIC-LARGE`
- **単眼深度のみ**: `depth-anything/DA3MONO-LARGE`
- **すべての機能**: `depth-anything/DA3NESTED-GIANT-LARGE`

### Q: CPUでも動作しますか？

A: はい、`--device cpu` を指定することで動作しますが、GPUよりも大幅に遅くなります。

### Q: カスタムデータセットで学習できますか？

A: これらのサンプルは推論専用です。学習方法についてはメインのREADMEを参照してください。

### Q: Windows で実行する際の注意点は？

A: すべてのサンプルはWindowsとLinuxの両方で動作するように設計されています。PowerShellまたはコマンドプロンプトで実行してください。パスの区切り文字は自動的に処理されます。

```powershell
# Windows (PowerShell) での実行例
python 01_single_image_inference.py --input C:\Users\YourName\Pictures\image.jpg --output .\output
```

### Q: 結果の精度を向上させるには？

A: 以下の方法を試してください：

1. より大きいモデルを使用（DA3-GIANT）
2. 処理解像度を上げる（`--process-res 672`）
3. 高品質な入力画像を使用
4. マルチビューの場合、より多くの視点から撮影

## 📝 ライセンス

これらのサンプルコードは、Depth Anything 3プロジェクトのライセンスに従います。詳細はプロジェクトのルートディレクトリにあるLICENSEファイルを参照してください。

## 🤝 貢献

バグ報告や機能リクエストは、GitHubのIssuesで受け付けています。

## 📚 参考資料

- [メインドキュメント](../README.md)
- [API リファレンス](../docs/API.md)
- [CLI リファレンス](../docs/CLI.md)
- [Hugging Face Hub](https://huggingface.co/depth-anything)

---

**Happy Depth Estimation! 🎉**
