# Family Tree Visualization Tool

## 概述

這個工具可以從 SAM2 tracking 結果中可視化 family trees（物件階層關係）。對於每個有 children 的 family，工具會找到最佳的 frame（family 成員同時出現最多的 frame），並生成包含原圖和各層級（L2/L4/L6）**segment cutouts** 的並排展示圖。

## 功能特點

- ✅ **自動找最佳 Frame**：選擇 family 成員（parent + children）同時出現最多的 frame
- ✅ **多層級並排展示**：原圖 + L2 + L4 + L6 並列展示
- ✅ **Segment Cutout 模式**：只顯示 mask 區域內的像素（白色背景），清晰展示物體形狀
- ✅ **彩色邊框標識**：Parent 使用紅色邊框，children 使用不同顏色邊框（綠、藍、黃等）
- ✅ **Family 信息標注**：顯示 level、object ID、children 數量和 frame 編號
- ✅ **支持大規模處理**：可批量處理所有 families

## 使用方法

### 基本用法

```bash
python scripts/visualize_family_trees.py <scene_root>
```

例如：

```bash
PYTHONPATH=src python scripts/visualize_family_trees.py \
    outputs/experiments/test_1024_1/scene_00005_01
```

### 參數說明

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `scene_root` | Scene 輸出根目錄（必需） | - |
| `--data-path` | Color frames 目錄路徑 | 自動從 scene 名稱推斷 |
| `--output-dir` | 可視化輸出目錄 | `{scene_root}/family_viz` |
| `--min-children` | 最少 children 數量過濾 | 1 |
| `--max-families` | 限制處理的 family 數量 | 全部 |

### 使用示例

**1. 處理所有 families：**

```bash
PYTHONPATH=src python scripts/visualize_family_trees.py \
    outputs/experiments/test_1024_1/scene_00005_01
```

**2. 只處理前 5 個 families（快速測試）：**

```bash
PYTHONPATH=src python scripts/visualize_family_trees.py \
    outputs/experiments/test_1024_1/scene_00005_01 \
    --max-families 5
```

**3. 只處理有至少 2 個 children 的 families：**

```bash
PYTHONPATH=src python scripts/visualize_family_trees.py \
    outputs/experiments/test_1024_1/scene_00005_01 \
    --min-children 2
```

**4. 指定自定義輸出目錄：**

```bash
PYTHONPATH=src python scripts/visualize_family_trees.py \
    outputs/experiments/test_1024_1/scene_00005_01 \
    --output-dir custom_viz
```

## 輸出格式

每個 family 會生成一個 JPEG 圖片，文件名格式：

```
family_L{level}_obj{obj_id:04d}_frame{frame_idx:06d}.jpg
```

例如：
- `family_L2_obj0018_frame000900.jpg` - Level 2, Object #18, Frame 900
- `family_L4_obj0438_frame004100.jpg` - Level 4, Object #438, Frame 4100

### 圖片布局

```
┌────────────┬────────────┬────────────┬────────────┐
│  Original  │     L2     │     L4     │     L6     │
├────────────┼────────────┼────────────┼────────────┤
│            │  ┌──┐      │    ┌───┐   │            │
│   原圖      │  │  │紅框  │    │   │   │  細節物體  │
│            │  └──┘      │    └───┘   │  (cutout)  │
│            │  白色背景    │   彩色邊框  │            │
└────────────┴────────────┴────────────┴────────────┘
Family: L2 Obj#18 → 1 children, 2 descendants | Frame: 900
```

- **Original**：原始 RGB 圖像（完整 frame）
- **L2/L4/L6**：對應層級的 **segment cutout** 圖
  - **黑色背景**：非 mask 區域（便於查看物體）
  - **原圖像素**：mask 區域內保留原始顏色和紋理
  - **紅色邊框**：Parent object（3 像素寬）
  - **彩色邊框**：Children objects（綠、藍、黃等，3 像素寬）

**Segment Cutout 的優點：**
- ✅ 清晰展示物體的精確形狀和邊界
- ✅ 黑色背景讓物體更突出，避免視覺干擾
- ✅ 容易辨識 parent 和 children 的空間關係
- ✅ 適合檢查 segmentation 質量和邊界準確性

## 實際案例

### Scene 00005_01 結果

在 `test_1024_1/scene_00005_01` 中找到了 **28 個 families**：

- **15 個** Level 2 families（粗粒度物體）
- **13 個** Level 4 families（中粒度物體）
- **0 個** Level 6 families（細粒度物體沒有 children）

每個 family 的可視化圖片大小約 **1.3-1.7 MB**，總共約 **40 MB**。

### 執行輸出示例

```
📂 Scene root: outputs/experiments/test_1024_1/scene_00005_01
📂 Data path: /media/public_dataset2/multiscan/scene_00005_01/outputs/color
📂 Output dir: outputs/experiments/test_1024_1/scene_00005_01/family_viz

🌲 Loading SAM2 tree...
Found 28 families with children
After filtering (min_children=1): 28 families

📊 Loading object-frame mappings...
  L2: 288 objects, linked video: video_segments_scale0.3x.npz
  L4: 656 objects, linked video: video_segments_scale0.3x.npz
  L6: 997 objects, linked video: video_segments_scale0.3x.npz

🎨 Generating visualizations for 28 families...

[1/28] Visualizing Family L2 Obj#18 (1 children) at frame 900...
✓ Saved: family_L2_obj0018_frame000900.jpg
[2/28] Visualizing Family L2 Obj#133 (4 children) at frame 2050...
✓ Saved: family_L2_obj0133_frame002050.jpg
...
[28/28] Visualizing Family L4 Obj#656 (1 children) at frame 7050...
✓ Saved: family_L4_obj0656_frame007050.jpg

✅ Done! Visualizations saved to: .../family_viz
```

## 技術細節

### Frame 選擇算法

對於每個 family，工具使用以下策略選擇最佳 frame：

1. 獲取 parent object 出現的所有 frames
2. 對每個 frame 計算 score = parent 出現（1） + 所有同時出現的 children 數量
3. 選擇 score 最高的 frame
4. 這確保可視化的 frame 展示了盡可能多的 family 成員

### Color Coding

工具使用預定義的顏色調色板：

```python
colors = [
    '#FF0000',  # Red (parent)
    '#00FF00',  # Green (child 1)
    '#0000FF',  # Blue (child 2)
    '#FFFF00',  # Yellow (child 3)
    '#FF00FF',  # Magenta (child 4)
    ...
]
```

如果 children 數量超過調色板大小，會自動生成 HSV 空間均勻分布的額外顏色。

### 依賴的文件

工具需要以下文件存在：

1. **`{scene_root}/relations/sam2_tree.json`** - SAM2 tree 結構
2. **`{scene_root}/level_{L}/object_segments_L{L:02d}.npz`** - Object-frame 映射（manifest）
3. **`{scene_root}/level_{L}/video_segments_L{L:02d}.npz`** - 實際 mask 數據
4. **`{data_path}/*.png` or `*.jpg`** - 原始 RGB frames

## 故障排除

### 問題：找不到 data_path

**錯誤信息：**
```
❌ Could not infer data_path. Please specify --data-path
```

**解決方法：**
手動指定 data path：
```bash
python scripts/visualize_family_trees.py \
    outputs/experiments/test_1024_1/scene_00005_01 \
    --data-path /path/to/multiscan/scene_00005_01/outputs/color
```

### 問題：Video NPZ not found

**錯誤信息：**
```
⚠ Video segments not found for L2
```

**可能原因：**
1. Tracking stage 未完成
2. NPZ 文件名不匹配（manifest 指向 `video_segments_scale0.3x.npz` 但實際是 `video_segments_L02.npz`）

**解決方法：**
工具會自動嘗試兩種命名方式，如果仍然失敗，請檢查 level 目錄下是否有 video_segments NPZ 文件。

### 問題：No frames found for family

**錯誤信息：**
```
[X/Y] ⚠ Family L2 Obj#123: No frames found
```

**可能原因：**
Object manifest 中沒有該 object 的 frame 記錄。

**解決方法：**
這是正常現象，跳過該 family 即可。可能是 tracking 過程中該 object 未被成功追蹤。

## 性能

- **處理速度**：約 1-2 families/秒（取決於 frame 大小和 mask 數量）
- **內存使用**：約 500-1000 MB（主要用於圖像處理）
- **輸出大小**：每個可視化圖片約 1-2 MB

對於包含 28 個 families 的場景：
- **總時間**：約 30-60 秒
- **輸出大小**：約 40-50 MB

## 限制

1. **只可視化有 children 的 families** - 孤立的 objects (orphans) 不會被可視化
2. **Frame 必須在 data_path 中存在** - 如果 manifest 中的 frame index 在實際數據中不存在，會跳過該 family
3. **需要完整的 tracking 輸出** - 必須有 object_segments 和 video_segments NPZ 文件

## 未來改進

- [ ] 支持自定義顏色方案
- [ ] 添加 mask 邊界描邊以提高可見度
- [ ] 支持生成 GIF 動畫展示 family 隨時間的演化
- [ ] 添加 HTML 報告生成，展示所有 families 的縮略圖
- [ ] 支持跨 level 的 parent-child 關係可視化

## 相關文檔

- [SAM2 Tree Building Guide](../docs/SAM2_TREE_USAGE.md)
- [Tree Algorithm Analysis](../docs/TREE_ALGORITHM_ANALYSIS.md)
- [Format Standards](../docs/FORMAT_STANDARDS_v3.md)
