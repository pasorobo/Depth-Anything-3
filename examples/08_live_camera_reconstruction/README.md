# 08. ライブカメラ3D再構成

USBカメラやウェブカメラからリアルタイムで3D再構成を行います。手動または自動でフレームをキャプチャし、点群を蓄積していきます。

## 🚀 クイックスタート

```bash
python live_camera_reconstruction.py --output ./live_reconstruction
```

## ⌨️ キーボード操作

| キー | 機能 |
|------|------|
| `SPACE` | 現在のフレームをキャプチャして再構成に追加 |
| `a` | 自動キャプチャモードの切り替え |
| `s` | 現在の再構成を保存 |
| `c` | 再構成をクリア（リセット） |
| `q` または `ESC` | 終了 |

## 📋 使用方法

### 基本的な使用

```bash
# デフォルトカメラを使用
python live_camera_reconstruction.py --output ./live_reconstruction

# 特定のカメラデバイスを指定（カメラID: 0, 1, 2...）
python live_camera_reconstruction.py --camera 1

# 自動キャプチャモード（30フレームごと）
python live_camera_reconstruction.py --auto-capture --capture-interval 30

# 深度マップをリアルタイム表示
python live_camera_reconstruction.py --show-depth

# カメラ解像度を指定
python live_camera_reconstruction.py --camera-width 1280 --camera-height 720
```

## ⚙️ オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--camera`, `-c` | カメラデバイスID | `0` |
| `--output`, `-o` | 出力ディレクトリ | `./live_reconstruction` |
| `--model`, `-m` | モデル名 | `depth-anything/DA3-LARGE` |
| `--process-res` | 処理解像度 | `504` |
| `--device` | 使用デバイス (cuda/cpu) | 自動検出 |
| `--camera-width` | カメラキャプチャ幅 | `640` |
| `--camera-height` | カメラキャプチャ高さ | `480` |
| `--auto-capture` | 自動キャプチャを有効化 | False |
| `--capture-interval` | 自動キャプチャ間隔（フレーム数） | `30` |
| `--max-frames` | 最大キャプチャフレーム数 | `100` |
| `--min-confidence` | 点の最小信頼度閾値 | `0.5` |
| `--show-depth` | 深度マップを別ウィンドウに表示 | False |

## 📤 出力

- `live_reconstruction_{timestamp}.ply` - PLY形式の点群
- `live_reconstruction_{timestamp}.npz` - NPZ形式のデータ

## 💡 使用例

### 例1: 手動スキャン（物体スキャン）

```bash
python live_camera_reconstruction.py --output ./object_scan

# 操作:
# 1. カメラを構えて物体を画面中央に配置
# 2. SPACEキーを押してキャプチャ
# 3. カメラをゆっくり動かして異なる角度から
# 4. 10-20枚キャプチャしたら 's' で保存
```

### 例2: 自動スキャン（部屋スキャン）

```bash
python live_camera_reconstruction.py \
    --auto-capture \
    --capture-interval 30 \
    --max-frames 50 \
    --show-depth \
    --output ./room_scan

# 操作:
# 1. カメラを持って部屋を歩く
# 2. 自動的にフレームがキャプチャされる
# 3. 50フレーム到達したら自動停止
```

### 例3: 高解像度スキャン

```bash
python live_camera_reconstruction.py \
    --camera-width 1920 \
    --camera-height 1080 \
    --process-res 672 \
    --model depth-anything/DA3-GIANT \
    --output ./high_res_scan
```

## 🎯 実用的な使用シーン

### 物体スキャン
1. 物体を回転台に配置
2. カメラを固定して手動キャプチャ
3. 10-15度ずつ回転させながらSPACEキー
4. 24-36枚で一周完了

### 部屋の3Dマップ作成
1. 自動キャプチャモードで開始
2. 部屋の中を歩き回る
3. すべての壁、家具を含める
4. 's'キーで保存、'q'で終了

### インテリアデザイン検討
1. 部屋の現状をスキャン
2. 3Dモデルとして保存
3. 別のソフトウェアで家具配置をシミュレーション

## 📺 画面表示の見方

ライブビューには以下の情報が表示されます：

```
┌─────────────────────────────┐
│ Frames captured: 15         │  # キャプチャ済みフレーム数
│ Total points: 1,234,567     │  # 累積点数
│ Inference: 45ms (22.2 FPS)  │  # 処理時間とFPS
│ Mode: AUTO                  │  # 手動/自動モード
├─────────────────────────────┤
│                             │
│    [カメラ映像]              │
│                             │
├─────────────────────────────┤
│ SPACE:Capture | a:Auto |    │  # キー操作ガイド
│ s:Save | c:Clear | q:Quit   │
└─────────────────────────────┘
```

## 🎨 スキャンのコツ

### 良い結果を得るために

1. **ゆっくり動く**: 1秒に10cm以下の移動速度
2. **重複を確保**: 前のフレームと70%以上重複
3. **一定の照明**: 明るさの変化を避ける
4. **テクスチャ重視**: 特徴的な模様がある場所を優先
5. **安定したカメラ**: 手ブレを最小限に

### 避けるべきこと

- ❌ 急激なカメラ移動
- ❌ 照明の急変（窓の前を通過など）
- ❌ 反射の強い表面（鏡、ガラス）
- ❌ 単一色の平坦な壁

## 🔧 トラブルシューティング

### カメラが認識されない

```bash
# 異なるカメラIDを試す
python live_camera_reconstruction.py --camera 1
python live_camera_reconstruction.py --camera 2

# Linuxの場合、デバイスを確認
ls /dev/video*
```

### 深度推定が遅い

```bash
# 処理解像度を下げる
python live_camera_reconstruction.py --process-res 336

# カメラ解像度を下げる
python live_camera_reconstruction.py --camera-width 640 --camera-height 480

# 軽量モデルを使用
python live_camera_reconstruction.py --model depth-anything/DA3-BASE
```

### 点群が大きすぎる

```bash
# 信頼度閾値を上げる
python live_camera_reconstruction.py --min-confidence 0.7

# 最大フレーム数を制限
python live_camera_reconstruction.py --max-frames 30
```

## 📊 パフォーマンス目安

| GPU | 処理解像度 | FPS（推論） | 推奨モード |
|-----|-----------|------------|----------|
| RTX 3090 | 504 | 20-25 | 自動キャプチャ |
| RTX 3060 | 504 | 10-15 | 手動キャプチャ |
| GTX 1660 | 336 | 8-12 | 手動キャプチャ |
| CPU only | 336 | 1-2 | 手動キャプチャ |

## 📚 関連サンプル

- [07. 大規模3D再構成](../07_advanced_3d_reconstruction/) - 動画からの大規模再構成
- [06. マルチビュー深度とカメラポーズ推定](../06_multiview_pose_estimation/) - 静止画からの再構成
- [ユーティリティツール](../utils/) - 点群の後処理と統合

## 🎓 チュートリアル動画（推奨）

このサンプルの使い方を学ぶには、以下の流れで練習してください：

1. **初心者**: 小さな物体を手動スキャン（10フレーム）
2. **中級者**: 机の上を自動スキャン（30フレーム）
3. **上級者**: 部屋全体をスキャン（50-100フレーム）

各レベルで保存した点群を[ユーティリティツール](../utils/)で統合・最適化してみましょう！
