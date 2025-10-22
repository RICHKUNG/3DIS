# My3DIS Workflow Status Report (2025-10-22)

> **Last Updated:** 2025-10-22
> **Git Commit:** fb2c3b4 (綜合大優化，整理檔案架構)
> **Status:** Production-ready with recent optimizations

## 執行摘要

My3DIS 工作流程已完成一系列重要優化和錯誤修復。當前系統支持：
- ✅ 多場景並行執行（帶 OOM 保護）
- ✅ 可配置的遮罩數量上限（包括無限制選項）
- ✅ 完整的家族關係追蹤（修復 cross-level 關係錯誤）
- ✅ 統一的實驗報告生成（check.md 整合）
- ✅ 彈性的配置系統（YAML + CLI + 環境變數）

---

## 一、當前工作流程架構

### 1.1 雙階段處理流程

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Semantic-SAM 候選生成                          │
│  ├─ Progressive refinement (多層級遮罩生成)              │
│  ├─ Gap filling (填補未覆蓋區域)                         │
│  ├─ Filtering (面積和穩定性過濾)                         │
│  └─ Output: raw/*.tar + filtered.json + tree.json       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Stage 2: SAM2 時序追蹤                                  │
│  ├─ Prompt generation (從候選生成提示)                   │
│  ├─ Temporal propagation (跨幀追蹤)                     │
│  ├─ Deduplication (IoU 去重)                            │
│  └─ Output: video_segments.npz + object_segments.npz    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Stage 3: 報告生成                                       │
│  ├─ Timing statistics                                   │
│  ├─ Visualization samples                               │
│  ├─ Cross-level relation index (relations.json)        │
│  └─ Experiment-level summary (check.md)                │
└─────────────────────────────────────────────────────────┘
```

### 1.2 多場景執行模式

**模式 A：順序執行**
```yaml
experiment:
  parallel_scenes: 1
```
- 一次處理一個場景
- 最安全，記憶體佔用最小
- 適合記憶體有限或調試環境

**模式 B：並行執行（帶 OOM 保護）**
```yaml
experiment:
  parallel_scenes: 2
  force_parallel: false  # 默認值
```
- 同時處理多個場景
- 自動監控 cgroup 記憶體事件
- OOM 監控不可用時自動降級到順序執行
- **推薦用於生產環境**

**模式 C：強制並行執行**
```yaml
experiment:
  parallel_scenes: 2
  force_parallel: true
```
- 無論 OOM 監控是否可用都強制並行
- 適合信任的環境或記憶體充足時
- 速度最快，但無 OOM 保護

---

## 二、最近優化與修復 (2025-10-20 至 2025-10-22)

### 2.1 關鍵錯誤修復

#### A. 家族關係計算錯誤 (2025-10-22)
**問題：** `relation_index.py` 中的 `build_cross_level_relations()` 只複製了 intra-level 子節點，沒有建立 cross-level 父子關係，導致所有場景的 family 數量顯示為 0。

**修復：** `src/my3dis/relation_index.py:309-329`
```python
# Build cross-level children relationships
for level in sorted(levels, reverse=True):
    if level not in hierarchy:
        continue
    for obj_id, node in hierarchy[level].items():
        parent_id = node.get('parent')
        if parent_id is None:
            continue
        # Find parent's level and add this object as its child
        for parent_level in sorted(levels):
            if parent_level >= level:
                break
            if parent_level in hierarchy and parent_id in hierarchy[parent_level]:
                parent_node = hierarchy[parent_level][parent_id]
                if obj_id not in parent_node['children']:
                    parent_node['children'].append(obj_id)
                break
```

**驗證：** scene_00005_00 現在正確顯示 125 families（修復前為 0）

#### B. Tree.json 生成失敗 (2025-10-21)
**問題：** `generate_candidates.py` 沒有傳遞 `save_root` 參數，且 `ssam_progressive_adapter.py` 要求 `persist_outputs=True` 才保存關係。

**修復：**
1. `src/my3dis/generate_candidates.py:638-642` - 新增參數：
   ```python
   progressive_iter = generate_with_progressive(
       # ... existing params ...
       save_root=run_root,
       persist_outputs=False,
       save_relations=True,
   )
   ```

2. `src/my3dis/ssam_progressive_adapter.py:345, 351` - 移除 `persist_outputs` 依賴：
   ```python
   # Before: if save_relations and persist_outputs and save_root is not None:
   # After:
   if save_relations and save_root is not None:
   ```

#### C. OOM 監控模組無法導入 (2025-10-22)
**問題：** `run_workflow.py` 只把 `src/` 加入 `sys.path`，但 `oom_monitor` 在項目根目錄，導致導入失敗。

**修復：** `src/my3dis/run_workflow.py:13-15`
```python
# Add project_root for oom_monitor module
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
```

**影響：** OOM 監控現在可以正常工作，支持並行執行的記憶體保護

#### D. max_masks_per_level: null 導致 TypeError (2025-10-22)
**問題：** YAML 中設置 `max_masks_per_level: null` 時，`int(None)` 拋出 TypeError。

**修復：** `src/my3dis/workflow/stage_config.py:127-143`
```python
max_masks_cfg = stage_cfg.get('max_masks_per_level')
if max_masks_cfg is None:
    max_masks_per_level = 2000  # Default value
else:
    max_masks_per_level = int(max_masks_cfg)
    # Allow 0 or -1 to represent unlimited
    if max_masks_per_level > 0:
        max_masks_per_level = max(1, max_masks_per_level)
    elif max_masks_per_level < 0:
        max_masks_per_level = 0  # Normalize to 0
```

### 2.2 新功能實現

#### A. 無限制遮罩上限支持 (2025-10-22)
**用途：** 允許在高解析度或複雜場景中生成所有可能的遮罩，不受數量限制。

**配置：**
```yaml
stages:
  ssam:
    max_masks_per_level: 0   # 或 -1，表示無上限
```

**實現位置：**
- `src/my3dis/workflow/stage_config.py:139-143` - 配置解析
- `src/my3dis/semantic_refinement.py:445, 588, 628` - 三處限制檢查

**邏輯：**
```python
# Only apply limit if max_masks_per_level > 0
if max_masks_per_level > 0 and len(masks) > max_masks_per_level:
    masks = masks[:max_masks_per_level]
```

#### B. 強制並行執行選項 (2025-10-22)
**用途：** 在 OOM 監控不可用時仍允許並行執行（類似 2025-10-08 前的行為）。

**配置：**
```yaml
experiment:
  parallel_scenes: 2
  force_parallel: true  # 繞過 OOM 檢查
```

**實現：** `src/my3dis/workflow/executor.py:399-413`
```python
force_parallel = bool(experiment_cfg.get('force_parallel', False))

if parallel_scenes > 1 and memory_event_paths is not None and not memory_readers:
    if force_parallel:
        LOGGER.warning(
            "Parallel execution enabled without OOM monitoring (force_parallel=true). "
            "Note: Out-of-memory events will not be detected."
        )
    else:
        LOGGER.warning("Parallel execution requested but OOM monitoring is unavailable; ...")
        parallel_scenes = 1
```

#### C. Check.md 實驗報告整合 (2025-10-21)
**用途：** 在實驗級別自動生成統計摘要，包含場景計時和輸出統計。

**配置：**
```yaml
stages:
  report:
    enabled: true
    generate_check_report: true
```

**輸出範例：** `outputs/experiments/{name}/check.md`
```markdown
# Experiment Report: v2_246_ssam2_filter2k_fill10k_prop10_iou06_ds03

## Stage Timings

| Scene | Total | ssam | filter | tracker | report |
| --- | --- | --- | --- | --- | --- |
| scene_00005_00 | 01:02:58 | 20:21 | 00:02 | 42:35 | 00:00 |
| scene_00005_01 | 57:42 | 17:39 | 00:02 | 40:01 | 00:00 |

## Output Statistics

| Scene | Total_Objs | Families | Orphans | Max_Depth | L2_Objs | L4_Objs | L6_Objs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scene_00005_00 | 512 | 125 | 89 | 3 | 212 | 295 | 5 |
| scene_00005_01 | 471 | 0 | 197 | 3 | 196 | 257 | 18 |
```

**實現模組：**
- `src/my3dis/workflow/reporting.py` - 新模組，統一報告邏輯
- `src/my3dis/workflow/scene_workflow.py:349-361` - 整合點

---

## 三、配置系統詳解

### 3.1 配置優先級

```
CLI 參數 > 環境變數 (MY3DIS_*) > YAML 配置 > 代碼默認值
```

### 3.2 關鍵配置參數

#### Experiment 層級
```yaml
experiment:
  name: experiment_name               # 實驗名稱（必填）
  dataset_root: /path/to/multiscan    # 數據集根目錄
  scenes: null                        # null=使用 scene_start/end, 'all'=所有場景, [...]=明確列表
  scene_start: scene_00005_00         # 起始場景（含）
  scene_end: scene_00098_02           # 結束場景（含）
  parallel_scenes: 2                  # 並行場景數量
  force_parallel: false               # 強制並行（繞過 OOM 檢查）
  levels: [2, 4, 6]                   # Progressive refinement 層級
  frames:
    step: 50                          # 幀採樣步長
  output_root: /path/to/outputs/{name}
  output_layout: scene_level          # scene_level | aggregate
```

#### SSAM 階段
```yaml
stages:
  gpu: 1                              # GPU ID（全域默認）
  ssam:
    enabled: true
    ssam_freq: 2                      # 每 N 層執行一次 SSAM
    min_area: 500                     # 最小遮罩面積（像素）
    stability_threshold: 1.0          # 穩定性閾值 [0-1]
    fill_area: 10000                  # Gap fill 最小面積
    max_masks_per_level: 2000         # 每層最大遮罩數（0/-1=無限制）
    downscale_masks: true             # 下採樣遮罩以節省空間
    downscale_ratio: 0.3              # 下採樣比例
    persist_raw: true                 # 保存原始候選
    skip_filtering: false             # 跳過內建過濾
    add_gaps: true                    # 啟用 gap filling
```

#### Tracker 階段
```yaml
stages:
  tracker:
    enabled: true
    prompt_mode: all_mask             # all_mask | box | point
    max_propagate: 30                 # 最大傳播幀數
    iou_threshold: 0.6                # 去重 IoU 閾值
    downscale_masks: true
    downscale_ratio: 0.3
    render_viz: true                  # 生成對比可視化
    sam2_cfg: /path/to/sam2.1_hiera_l.yaml
    sam2_ckpt: /path/to/sam2.1_hiera_large.pt
    comparison_sampling:
      max_frames: 300                 # 可視化最大幀數
```

#### Report 階段
```yaml
stages:
  report:
    enabled: true
    name: report.md
    max_width: 640                    # 預覽圖最大寬度
    record_timings: true
    timing_output: stage_timings.json
    generate_check_report: true       # 生成實驗級摘要
```

### 3.3 環境變數覆蓋

所有外部依賴路徑都可以通過環境變數覆蓋（優先級最高）：

```bash
export MY3DIS_SEMANTIC_SAM_ROOT=/path/to/Semantic-SAM
export MY3DIS_SEMANTIC_SAM_CKPT=/path/to/ckpt.pth
export MY3DIS_SAM2_ROOT=/path/to/SAM2
export MY3DIS_SAM2_CFG=/path/to/sam2.1_hiera_l.yaml
export MY3DIS_SAM2_CKPT=/path/to/sam2.1_hiera_large.pt
export MY3DIS_DATASET_ROOT=/path/to/multiscan
export MY3DIS_OUTPUT_ROOT=/path/to/outputs
```

**參考：** `src/my3dis/pipeline_defaults.py:9`

---

## 四、已知問題與限制

### 4.1 環境依賴

**雙 Conda 環境需求：**
- **Semantic-SAM 環境：** Detectron2 0.6 + PyTorch 1.13
- **SAM2 環境：** PyTorch 2.x

**當前解決方案：**
- `run_experiment.sh` 自動切換環境
- YAML 工作流程使用單一環境（需要手動確保兼容性）

**潛在問題：** 版本衝突導致無法在單一環境中運行完整流程

### 4.2 OOM 監控限制

**依賴項：**
- 項目本地模組 `oom_monitor/`（不是 PyPI 套件）
- Linux cgroup v1 或 v2 支持
- 可訪問的 `memory.events` 或 `memory.oom_control` 文件

**失敗模式：**
- 如果 OOM 監控不可用且 `force_parallel=false`，自動降級到順序執行
- 如果 `force_parallel=true`，繞過檢查但無 OOM 保護

**建議：**
- 生產環境建議啟用 OOM 監控
- 測試環境可使用 `force_parallel: true`

### 4.3 記憶體佔用

**高峰階段：**
1. **SSAM Progressive Refinement** - 每層級生成時
2. **SAM2 Tracking** - 載入 SAM2 模型時（~8GB GPU）
3. **並行執行** - 多場景同時運行時

**緩解策略：**
- 使用 `downscale_ratio: 0.3` 降低遮罩解析度
- 限制 `max_propagate` 減少追蹤幀數
- 調整 `parallel_scenes` 控制並發度
- 設置 `max_masks_per_level` 限制候選數量

### 4.4 磁碟 I/O

**瓶頸：**
- 原始候選存儲（chunked tar archives）
- NPZ/ZIP 檔案讀寫
- 可視化圖像生成

**優化：**
- 使用 `persist_raw: false` 跳過原始存儲（除非需要重新過濾）
- 設置 `render_viz: false` 禁用可視化（節省時間和空間）
- 使用 SSD 存儲輸出目錄

---

## 五、性能優化建議

### 5.1 速度優化

**場景處理速度：**
```
典型場景（200 幀，2/4/6 層級）：
- SSAM: 15-40 分鐘
- Filter: < 1 分鐘
- Tracker: 30-90 分鐘
- 總計: 45-130 分鐘/場景
```

**加速策略：**

1. **並行執行**
   ```yaml
   parallel_scenes: 2  # 根據 GPU 數量調整
   force_parallel: true  # 如果記憶體充足
   ```

2. **減少候選數量**
   ```yaml
   ssam:
     min_area: 1000      # 從 500 增加到 1000
     max_masks_per_level: 1000  # 從 2000 減少
   ```

3. **降低追蹤成本**
   ```yaml
   tracker:
     max_propagate: 10   # 從 30 減少
     render_viz: false   # 禁用可視化
   ```

4. **跳過報告階段（測試時）**
   ```yaml
   report:
     enabled: false
   ```

### 5.2 質量優化

**提升遮罩質量：**

1. **更密集的採樣**
   ```yaml
   frames:
     step: 20  # 從 50 減少到 20
   ```

2. **更嚴格的過濾**
   ```yaml
   ssam:
     min_area: 2000
     stability_threshold: 0.95
   ```

3. **更長的追蹤**
   ```yaml
   tracker:
     max_propagate: 50
     iou_threshold: 0.7
   ```

4. **無限制候選**
   ```yaml
   ssam:
     max_masks_per_level: 0  # 不限制數量
   ```

### 5.3 記憶體優化

**降低記憶體佔用：**

1. **順序執行**
   ```yaml
   parallel_scenes: 1
   ```

2. **更積極的下採樣**
   ```yaml
   downscale_ratio: 0.2  # 從 0.3 降低
   ```

3. **限制候選數量**
   ```yaml
   max_masks_per_level: 500
   ```

4. **不保存原始候選**
   ```yaml
   persist_raw: false
   ```

---

## 六、故障排除指南

### 6.1 常見錯誤

#### A. "Parallel execution requested but OOM monitoring is unavailable"

**原因：** OOM 監控無法初始化，但設置了 `parallel_scenes > 1`

**解決方案：**
1. **檢查 oom_monitor 模組：**
   ```bash
   python3 -c "import oom_monitor; print(oom_monitor.__file__)"
   ```

2. **確認 sys.path 包含項目根目錄：**
   - 確認 `src/my3dis/run_workflow.py:13-15` 的修復已應用

3. **使用 force_parallel：**
   ```yaml
   experiment:
     force_parallel: true
   ```

#### B. "invalid stages.ssam.max_masks_per_level: None"

**原因：** YAML 中設置 `max_masks_per_level: null`

**解決方案：**
1. 設置具體數值：`max_masks_per_level: 2000`
2. 或使用無限制：`max_masks_per_level: 0`
3. 或移除該行（使用默認值 2000）

#### C. Family 數量顯示為 0

**原因：** 使用了舊版本的 `relation_index.py`（2025-10-22 前）

**解決方案：**
1. 更新到最新版本（包含 cross-level 關係修復）
2. 重新運行報告階段：
   ```bash
   python3 src/my3dis/generate_report.py --run-dir <path>
   ```

#### D. Tree.json 未生成

**原因：** 缺少 `save_root` 或 `save_relations` 參數

**解決方案：**
1. 確認使用最新版本的 `generate_candidates.py`
2. 檢查 `ssam_progressive_adapter.py` 已移除 `persist_outputs` 依賴

### 6.2 性能診斷

**檢查瓶頸：**

1. **查看階段計時：**
   ```bash
   cat outputs/experiments/{name}/{scene}/stage_timings.json
   ```

2. **檢查記憶體使用：**
   ```bash
   grep "peak_rss_mib" outputs/experiments/{name}/{scene}/workflow_summary.json
   ```

3. **查看 GPU 利用率：**
   ```bash
   nvidia-smi dmon -s u
   ```

4. **分析日誌：**
   ```bash
   tail -f logs/scene_workers/{run}/*.log
   ```

---

## 七、開發路線圖

### 7.1 短期目標（1-2 週）

- [ ] 單一 Conda 環境支持（解決 PyTorch 版本衝突）
- [ ] 自動化測試套件（單元測試 + 整合測試）
- [ ] GPU 記憶體監控整合到 OOM 監控
- [ ] 增量更新支持（只處理新幀）

### 7.2 中期目標（1-2 月）

- [ ] 分散式執行支持（多機器並行）
- [ ] 實時進度追蹤（Web UI）
- [ ] 自動超參數調優
- [ ] 更多數據集支持（ScanNet, Replica）

### 7.3 長期目標（3-6 月）

- [ ] 端到端神經網路替代 Progressive Refinement
- [ ] 在線學習和自適應追蹤
- [ ] 3D 重建整合
- [ ] 語義標註整合

---

## 八、參考文獻

### 8.1 內部文檔

- `CLAUDE.md` - 項目總覽和環境設置
- `src/my3dis/OVERVIEW.md` - 模組執行流程
- `src/my3dis/workflow/WORKFLOW_GUIDE.md` - 工作流程詳解
- `src/my3dis/tracking/TRACKING_GUIDE.md` - 追蹤系統內部
- `docs/QUICK_START_RECOVERY.md` - 關係恢復快速指南
- `docs/RELATION_INDEX.md` - 關係索引格式說明

### 8.2 配置範例

- `configs/multiscan/base.yaml` - 生產配置模板
- `configs/multiscan/test_65.yaml` - 測試配置

### 8.3 工具腳本

- `run_experiment.sh` - 雙環境執行腳本
- `scripts/batch_recover_v2_relations.py` - 批次關係恢復
- `scripts/check_scene_timings.py` - 計時統計工具

---

## 附錄 A：配置模板

### A.1 生產環境配置

```yaml
experiment:
  name: production_run
  dataset_root: /media/public_dataset2/multiscan
  scenes: all
  parallel_scenes: 2
  force_parallel: false  # 依賴 OOM 監控
  levels: [2, 4, 6]
  frames:
    step: 50
  output_root: /media/Pluto/richkung/My3DIS/outputs/experiments/{name}
  output_layout: scene_level

stages:
  gpu: 0
  ssam:
    enabled: true
    ssam_freq: 2
    min_area: 2000
    stability_threshold: 1.0
    fill_area: 10000
    max_masks_per_level: 2000
    downscale_ratio: 0.3

  tracker:
    enabled: true
    max_propagate: 10
    iou_threshold: 0.6
    downscale_ratio: 0.3
    render_viz: true

  report:
    enabled: true
    generate_check_report: true
```

### A.2 快速測試配置

```yaml
experiment:
  name: quick_test
  dataset_root: /media/public_dataset2/multiscan
  scene_start: scene_00005_00
  scene_end: scene_00005_01
  parallel_scenes: 2
  force_parallel: true  # 繞過 OOM 檢查
  levels: [2, 4]  # 只測試兩層
  frames:
    step: 100  # 更少幀數
  output_root: /tmp/my3dis_test/{name}

stages:
  gpu: 0
  ssam:
    enabled: true
    min_area: 500
    max_masks_per_level: 500  # 限制數量加速
    downscale_ratio: 0.2

  tracker:
    enabled: true
    max_propagate: 5  # 短追蹤
    render_viz: false  # 禁用可視化

  report:
    enabled: false  # 跳過報告
```

### A.3 高質量配置

```yaml
experiment:
  name: high_quality_run
  dataset_root: /media/public_dataset2/multiscan
  scenes: [scene_00065_00]  # 單場景精細處理
  parallel_scenes: 1
  levels: [2, 4, 6]
  frames:
    step: 20  # 密集採樣
  output_root: /media/Pluto/richkung/My3DIS/outputs/experiments/{name}

stages:
  gpu: 0
  ssam:
    enabled: true
    ssam_freq: 2
    min_area: 2000
    stability_threshold: 0.95
    fill_area: 5000
    max_masks_per_level: 0  # 無限制
    downscale_ratio: 0.5  # 更高解析度

  tracker:
    enabled: true
    max_propagate: 50  # 長期追蹤
    iou_threshold: 0.7
    downscale_ratio: 0.5
    render_viz: true
    comparison_sampling:
      max_frames: 500

  report:
    enabled: true
    max_width: 1024
    generate_check_report: true
```

---

## 附錄 B：版本歷史

### 2025-10-22
- ✅ 修復 OOM 監控導入問題
- ✅ 實現無限制 max_masks 支持
- ✅ 新增 force_parallel 選項
- ✅ 修復 family 關係計算錯誤

### 2025-10-21
- ✅ 整合 check.md 生成到 workflow
- ✅ 修復 tree.json 生成問題
- ✅ 新增 workflow/reporting.py 模組
- ✅ 統一實驗報告格式

### 2025-10-20
- ✅ 新增 check_scene_timings.py 輸出統計
- ✅ 創建 workflow_history.csv 符號連結
- ✅ 增強統計數據收集

### 2025-10-09
- ✅ 新增 OOM 監控支持
- ✅ 實現並行執行安全檢查

### 2025-10-08
- ✅ 重構工作流程執行和日誌系統
- ✅ 實現多場景並行執行

---

**文檔維護者：** Claude Code Assistant
**審查週期：** 每次重大更新後
**聯繫方式：** 參考項目 README
