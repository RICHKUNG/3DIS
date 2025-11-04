# 問題：如何確保每個 L4、L6 物件都有對應的 parent？

## 背景

以漸進式切割 L2、L4、L6 為例，目前的做法是：
1. **SSAM 漸進式切割**：產生各 level 的候選 masks
2. **SAM2 tracking**：用 SSAM masks 作為 prompt 進行時序追蹤
3. **事後計算親子關係**：計算 tracked objects 之間的 mask 包裹關係建立 family

**問題**：這樣產生的 family 比預期少。

## 原始想法

1. **在 SSAM 階段就建立 family**：依據 L2、L4、L6 mask 包裹關係建立 family（每個 SSAM frame 的每個 L2 object 都有一個 family）
2. **在 SAM2 tracking 時記錄 SSAM → SAM2 映射**：當 SAM2 deduplication 發現某個 SSAM mask 和已追蹤的 mask IoU > 門檻時，記錄這個「SSAM mask ID → SAM2 object ID」的對應關係

## 方案選擇

**✅ 採用：統一映射方案（Unified Mapping）**

核心設計：
- 使用單一字典 `ssam_to_sam2: Dict[str, int]` 記錄所有 SSAM → SAM2 映射
- Accepted prompts（新 tracking）和 rejected prompts（dedup 映射）統一處理
- 查找 parent 只需 O(1) 單次字典查找
- 代碼更簡潔、更易維護

**❌ 不採用：分離映射方案（Separated Mapping）**

原方案使用兩個字典：
- `accepted_map: Dict[str, int]` - 記錄 accepted prompts
- `rejected_map: Dict[str, int]` - 記錄 rejected prompts
- 查找時需要兩次查表（先查 accepted，再查 rejected）
- 概念上更複雜，無實質優勢

## 關鍵場景說明

**為什麼需要記錄 dedup rejection mapping？**

考慮以下場景：

```
Frame 50 (SSAM 首次處理):
├─ L2_A (ssam_id: 50_2_001) → SAM2 開始追蹤 → sam2_obj_id=1
└─ L4_A (ssam_id: 50_4_001, parent: 50_2_001) → SAM2 開始追蹤 → sam2_obj_id=2

SAM2 propagates: Frame 50 → 100

Frame 100 (SSAM 再次處理):
├─ L2_B (ssam_id: 100_2_001) → IoU(L2_B, sam2_obj_id=1) > threshold → ❌ 被 dedup 丟棄
├─ L4_B (ssam_id: 100_4_001, parent: 100_2_001) → IoU(L4_B, sam2_obj_id=2) > threshold → ❌ 被 dedup 丟棄
└─ L4_C (ssam_id: 100_4_002, parent: 100_2_001) → 新物件 → ✅ SAM2 開始追蹤 → sam2_obj_id=3
```

**問題出現**：
- `sam2_obj_id=3` (L4_C) 的 provenance 記錄 `ssam_parent_id=100_2_001`
- 但 `100_2_001` 這個 SSAM mask 被 dedup **丟棄**了，從未被 SAM2 追蹤
- 如果只用 provenance map 查找，會找不到 parent → **孤兒物件**

**解決方案**：
在 dedup rejection 時記錄：`ssam_to_sam2['0100_2_001'] = 1`

這樣查找 parent 時：
```python
# ✅ 使用統一映射，只需一次查表
parent_sam2_id = ssam_to_sam2.get(ssam_parent_id)

# 如果找不到：
# - 可能 parent 被 SSAM 產生但被 filter 過濾掉（min_area、stability）
# - 或者 parent 根本沒被 SSAM 產生
if parent_sam2_id is None:
    # 真正的孤兒
    ...
```

**優點**：
- ✅ 無需區分 accepted/rejected，統一處理
- ✅ 查找 parent 只需 O(1) 單次字典查找
- ✅ 代碼更簡潔、更易維護

## 當前方法的問題診斷

你現在的流程是:

1. **SSAM 階段**: 在每個 frame 獨立產生 L2/L4/L6 masks,此時已有「同 frame 內」的包裹關係
2. **SAM2 階段**: 用 SSAM masks 作為 prompt,追蹤產生「跨 frame」的物件
3. **事後分析**: 計算 SAM2 tracked objects 之間的包裹關係建立 family

問題出在**第三步驟**。當你事後分析 SAM2 結果時,可能遇到這些狀況:

- **時序不重疊**: L2 物件出現在 frame 1-10,但它的 L4 children 只在 frame 5-15 被追蹤到,導致共同出現的 frames 太少
- **追蹤漂移**: SAM2 可能把原本屬於同一個 SSAM family 的 L4 masks 追蹤成不同 object IDs
- **遮擋問題**: 當物件被遮擋時,包裹關係可能暫時消失,導致平均包裹率低於門檻

## 你的想法是對的方向

你提到的兩階段策略其實非常合理:

**階段一**: 在 SSAM 時就記錄「per-frame family」,這樣每個 L4 mask 在被創建時就知道它來自哪個 L2 parent。

**階段二**: 在 SAM2 追蹤時,當發現兩個 SSAM masks 其實是同一物件時,合併它們的 family。

這個想法的核心洞察是:**hierarchical relationship 應該在 mask 產生時就建立,而不是事後推斷**。

---

## 具體實作建議

### ⚠️ 核心問題：SSAM Unique ID 衝突

**當前實現的致命缺陷**（`semantic_refinement.py:419`）：
```python
mask_id_counter = 1  # ❌ 每次調用 progressive_refinement_masks 時重置
```

這導致不同 SSAM frame 產生的 mask 會有**相同的 unique_id**：
- Frame 50: L2 mask → unique_id=1
- Frame 100: L2 mask → unique_id=1（衝突！）

**解決方案：使用 Composite Key**

採用格式：`{ssam_frame_idx:04d}_{level}_{seq:04d}`
- 範例：`0050_2_0001` = Frame 50, Level 2, Sequence 1
- 範例：`0100_4_0023` = Frame 100, Level 4, Sequence 23

---

### 第一步：修復 SSAM Unique ID 生成

#### 1.1 修改 `semantic_refinement.py` 的 ID 生成邏輯

**目標文件**：`src/my3dis/semantic_refinement.py`

**修改位置 1**：函數簽名（Line 351）
```python
def progressive_refinement_masks(
    semantic_sam,
    image_path: str,
    level_sequence: List[int],
    output_dirs: Dict[str, str],
    *,
    ssam_frame_idx: int = 0,  # ✅ 新增：SSAM frame 索引
    min_area: int = 50,
    max_masks_per_level: int = 200,
    # ... 其他參數
) -> Dict[str, Any]:
```

**修改位置 2**：ID counter 初始化（Line 419）
```python
# ❌ 舊代碼
mask_id_counter = 1

# ✅ 新代碼
# 使用 dict 追蹤每個 level 的序號
level_seq_counters = {level: 1 for level in level_sequence}

def make_unique_id(level: int) -> str:
    """生成全局唯一的 composite ID"""
    seq = level_seq_counters[level]
    level_seq_counters[level] += 1
    return f"{ssam_frame_idx:04d}_{level}_{seq:04d}"
```

**修改位置 3**：First level ID 賦值（Line 440）
```python
# ❌ 舊代碼
mask["unique_id"] = mask_id_counter
mask_id_counter += 1

# ✅ 新代碼
mask["unique_id"] = make_unique_id(first_level)
mask["ssam_frame_idx"] = ssam_frame_idx
```

**修改位置 4**：Child mask ID 賦值（Line 573-574）
```python
# ❌ 舊代碼
child["parent_unique_id"] = current_masks[parent_idx]["unique_id"]
child["unique_id"] = mask_id_counter
mask_id_counter += 1

# ✅ 新代碼
parent_uid = current_masks[parent_idx]["unique_id"]
child_uid = make_unique_id(next_level)

child["parent_unique_id"] = parent_uid
child["unique_id"] = child_uid
child["ssam_frame_idx"] = ssam_frame_idx

# 可選：記錄完整的 lineage（祖先鏈）
parent_lineage = current_masks[parent_idx].get("lineage", [])
child["lineage"] = parent_lineage + [parent_uid]
```

#### 1.2 修改 `ssam_progressive_adapter.py` 傳遞 frame index

**目標文件**：`src/my3dis/ssam_progressive_adapter.py`

**修改位置**：調用 `progressive_refinement_masks`（Line 256-265）
```python
# ❌ 舊代碼
return progressive_refinement_masks(
    semantic_sam,
    image_path,
    level_sequence=levels,
    output_dirs=output_dirs,
    # ...
)

# ✅ 新代碼
return progressive_refinement_masks(
    semantic_sam,
    image_path,
    level_sequence=levels,
    output_dirs=output_dirs,
    ssam_frame_idx=f_idx,  # 傳遞當前 frame 索引
    # ...
)
```

---

### 第二步：在 Candidate 持久化時保留 Provenance

#### 2.1 修改 `generate_candidates.py` 保存 metadata

**目標文件**：`src/my3dis/generate_candidates.py`

**修改位置**：Filtered candidate persistence（Line 801-804）
```python
# ✅ 已經正確：meta = {k: v for k, v in m.items() if k != 'segmentation'}
# 這行會複製所有非 'segmentation' 的 key，包括：
# - unique_id
# - parent_unique_id
# - ssam_frame_idx
# - lineage（如果有）

# ⚠️ 確認這些欄位確實存在於 m 中即可
```

**驗證方法**：檢查 `filtered.json` 是否包含這些欄位。

---

### 第三步：SAM2 Tracking 時維護 Provenance Mapping

#### 3.1 創建 `ProvenanceTracker` class

**目標文件**：`src/my3dis/tracking/provenance_tracker.py`（新文件）

**設計理念**：使用統一的 SSAM → SAM2 映射，無需區分 accepted/rejected，簡化查找邏輯。

```python
from typing import Dict, Optional, Any

class ProvenanceTracker:
    """
    追蹤 SSAM candidates 到 SAM2 tracked objects 的映射關係（統一映射方案）。

    核心概念：
    - 每個 SSAM unique_id 對應到一個 SAM2 object ID
    - 無論該 SSAM mask 是被 accepted（新 tracking）還是 rejected（dedup 映射到已存在 object）
    - 只需一個統一的映射表，查找時一次查表即可

    核心功能：
    1. 記錄 SSAM → SAM2 映射（accepted 和 rejected 統一處理）
    2. 記錄 SAM2 objects 的完整 provenance metadata
    3. 查詢 parent 關係：ssam_parent_id → sam2_parent_obj_id（O(1) 單次查表）
    """

    def __init__(self):
        # ✅ 統一映射：SSAM ID → SAM2 object ID
        # 包含 accepted prompts 和 rejected prompts（dedup 映射）
        self.ssam_to_sam2: Dict[str, int] = {}

        # SAM2 object ID → complete provenance metadata
        # 注意：只有 accepted prompts 才會有 provenance（rejected 的不是新 object）
        self.sam2_provenance: Dict[int, Dict[str, Any]] = {}

    def register_accepted_prompt(
        self,
        sam2_obj_id: int,
        ssam_unique_id: str,
        ssam_parent_id: Optional[str],
        ssam_frame_idx: int,
        level: int,
        lineage: Optional[list] = None
    ):
        """
        當 SSAM candidate 被 SAM2 接受為新 prompt 時調用。

        這個 SSAM mask 開始被 SAM2 追蹤，產生一個新的 tracked object。
        """
        self.ssam_to_sam2[ssam_unique_id] = sam2_obj_id
        self.sam2_provenance[sam2_obj_id] = {
            "ssam_unique_id": ssam_unique_id,
            "ssam_parent_id": ssam_parent_id,
            "ssam_frame_idx": ssam_frame_idx,
            "level": level,
            "lineage": lineage or [],
        }

    def register_rejected_prompt(
        self,
        ssam_unique_id: str,
        matched_sam2_obj_id: int
    ):
        """
        當 SSAM candidate 因 dedup IoU > threshold 被拒絕時調用。

        這個 SSAM mask 不會開始新的 tracking，而是被識別為已存在的 SAM2 object。
        記錄這個映射關係，使得後續查找 parent 時可以找到對應的 SAM2 ID。

        注意：如果同一個 ssam_unique_id 已經記錄過，保留第一次遇到的映射。
        """
        if ssam_unique_id not in self.ssam_to_sam2:
            self.ssam_to_sam2[ssam_unique_id] = matched_sam2_obj_id

    def find_parent(self, sam2_obj_id: int) -> Optional[int]:
        """
        查找給定 SAM2 object 的 parent object ID。

        流程：
        1. 從 sam2_provenance 取得 ssam_parent_id
        2. 查 ssam_to_sam2 映射表，找到 parent 對應的 SAM2 ID
        3. 找不到 → None（孤兒）

        複雜度：O(1) 單次字典查找
        """
        prov = self.sam2_provenance.get(sam2_obj_id)
        if prov is None:
            return None

        ssam_parent_id = prov.get("ssam_parent_id")
        if ssam_parent_id is None:
            return None  # Root level object (e.g., L2)

        # ✅ 只需一次查找！無論 parent 是 accepted 還是 rejected 都能找到
        return self.ssam_to_sam2.get(ssam_parent_id)

    def get_family_hierarchy(self) -> Dict:
        """
        導出完整的 parent map: {child_sam2_id: parent_sam2_id}

        同時統計孤兒數量（有 ssam_parent_id 但找不到對應的 SAM2 parent）
        """
        parent_map = {}
        orphans = []

        for sam2_obj_id in self.sam2_provenance.keys():
            parent_id = self.find_parent(sam2_obj_id)
            parent_map[sam2_obj_id] = parent_id

            # 統計孤兒（child level 但沒 parent）
            prov = self.sam2_provenance[sam2_obj_id]
            if prov.get("ssam_parent_id") is not None and parent_id is None:
                orphans.append(sam2_obj_id)

        return {
            "parent_map": parent_map,
            "orphan_count": len(orphans),
            "orphan_ids": orphans,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        導出統計信息

        注意：
        - accepted_prompts = len(sam2_provenance)（只有 accepted 才有 provenance）
        - rejected_prompts = len(ssam_to_sam2) - len(sam2_provenance)
        """
        accepted_count = len(self.sam2_provenance)
        total_count = len(self.ssam_to_sam2)
        rejected_count = total_count - accepted_count

        return {
            "total_ssam_masks": total_count,
            "accepted_prompts": accepted_count,
            "rejected_prompts": rejected_count,
            "rejection_rate": rejected_count / total_count if total_count > 0 else 0.0,
        }
```

---

### 第四步：修改 DedupStore 返回 Rejection Information

#### 4.1 修改 `stores.py` 的 `filter_candidates` 方法

**目標文件**：`src/my3dis/tracking/stores.py`

**修改位置**：`DedupStore.filter_candidates` 方法（Line 139-153）

```python
# ❌ 舊代碼
def filter_candidates(
    self,
    frame_idx: int,
    candidates: List["PromptCandidate"],
    threshold: float,
) -> List["PromptCandidate"]:
    accepted: List["PromptCandidate"] = []
    for cand in candidates:
        seg = cand.seg_for_iou
        if seg is not None and self.has_overlap(frame_idx, seg, threshold):
            continue  # Rejected
        accepted.append(cand)
        if seg is not None:
            self.add_mask(frame_idx, seg)
    return accepted

# ✅ 新代碼
def filter_candidates(
    self,
    frame_idx: int,
    candidates: List["PromptCandidate"],
    threshold: float,
) -> Tuple[List["PromptCandidate"], List[Tuple["PromptCandidate", int]]]:
    """
    Filter candidates based on IoU deduplication.

    Returns:
        accepted: List of candidates that passed dedup
        rejected: List of (candidate, matched_mask_index) tuples for rejected candidates
    """
    accepted: List["PromptCandidate"] = []
    rejected: List[Tuple["PromptCandidate", int]] = []

    for cand in candidates:
        seg = cand.seg_for_iou
        if seg is not None:
            # 找出重疊的 mask index
            match_idx = self.find_overlapping_mask_index(frame_idx, seg, threshold)
            if match_idx is not None:
                rejected.append((cand, match_idx))
                continue

        accepted.append(cand)
        if seg is not None:
            self.add_mask(frame_idx, seg)

    return accepted, rejected

def find_overlapping_mask_index(
    self, frame_idx: int, mask: np.ndarray, threshold: float
) -> Optional[int]:
    """
    找出與給定 mask 重疊的已存在 mask 的 index。

    Returns:
        int: 重疊 mask 的 index (在該 frame 的 masks list 中的位置)
        None: 沒有重疊的 mask
    """
    entry = self._frames.get(frame_idx)
    if entry is None or not entry.masks:
        return None

    mask_bool = np.asarray(mask, dtype=np.bool_)
    resized = self._resize(mask_bool, entry.target_shape)

    existing_stack = np.stack(entry.masks, axis=0)
    cand_broadcast = resized[np.newaxis, :, :]

    inter = np.logical_and(existing_stack, cand_broadcast).sum(axis=(1, 2))
    union = np.logical_or(existing_stack, cand_broadcast).sum(axis=(1, 2))

    valid = union > 0
    if not valid.any():
        return None

    ious = np.zeros(len(entry.masks))
    ious[valid] = inter[valid].astype(float) / union[valid].astype(float)

    max_idx = int(ious.argmax())
    if ious[max_idx] > float(threshold):
        return max_idx

    return None
```

**⚠️ 注意**：這需要追蹤 mask index → sam2_obj_id 的映射。

---

### 第五步：整合 ProvenanceTracker 到 SAM2 Tracking

#### 5.1 修改 `sam2_runner.py` 的 tracking 函數

**目標文件**：`src/my3dis/tracking/sam2_runner.py`

**修改位置 1**：Import ProvenanceTracker
```python
from my3dis.tracking.provenance_tracker import ProvenanceTracker
```

**修改位置 2**：初始化 tracker（Line ~250）
```python
def sam2_tracking(
    # ... existing parameters ...
) -> TrackingArtifacts:

    # ✅ 新增 ProvenanceTracker
    provenance_tracker = ProvenanceTracker()

    # ✅ 追蹤 dedup store 中每個 mask 的來源（用於 rejection mapping）
    # Key: (frame_idx, mask_index_in_dedup_store) → Value: sam2_obj_id
    dedup_mask_to_sam2_id: Dict[Tuple[int, int], int] = {}
```

**修改位置 3**：記錄 accepted prompts（Line ~340-370）
```python
# 在 _add_prompts_to_predictor 之後
for idx, cand in enumerate(filtered_candidates):
    sam2_obj_id = obj_count_before + idx  # SAM2 分配的 object ID

    # ✅ 記錄 provenance（accepted prompt）
    payload = cand.payload  # 來自 filtered.json 的 metadata
    ssam_unique_id = payload.get("unique_id")

    if ssam_unique_id:
        provenance_tracker.register_accepted_prompt(
            sam2_obj_id=sam2_obj_id,
            ssam_unique_id=ssam_unique_id,
            ssam_parent_id=payload.get("parent_unique_id"),
            ssam_frame_idx=payload.get("ssam_frame_idx", frame_idx),
            level=payload.get("level"),
            lineage=payload.get("lineage"),
        )

    # ✅ 記錄 dedup store 中的 mask index 映射（用於後續 rejection 查找）
    # 這個 mask 現在加入了 dedup_store，記錄它的位置
    mask_idx_in_store = len(dedup_store._frames[frame_idx].masks) - len(filtered_candidates) + idx
    dedup_mask_to_sam2_id[(frame_idx, mask_idx_in_store)] = sam2_obj_id
```

**修改位置 4**：記錄 rejected prompts
```python
# 在 dedup_store.filter_candidates 調用之前，先保存當前 batch 的 candidates
candidate_payloads = {cand: cand.payload for cand in batch}

# 調用 dedup
accepted, rejected = dedup_store.filter_candidates(frame_idx, batch, iou_threshold)

# ✅ 記錄 rejections
for cand, matched_mask_idx in rejected:
    payload = candidate_payloads[cand]
    ssam_unique_id = payload.get("unique_id")

    if ssam_unique_id:
        # 找出這個 matched_mask_idx 對應的 sam2_obj_id
        matched_sam2_id = dedup_mask_to_sam2_id.get((frame_idx, matched_mask_idx))

        if matched_sam2_id is not None:
            provenance_tracker.register_rejected_prompt(
                ssam_unique_id=ssam_unique_id,
                matched_sam2_obj_id=matched_sam2_id,
            )
```

**修改位置 5**：導出 provenance-based tree（最後）
```python
# ✅ 在 tracking 完成後，構建 family hierarchy
family_result = provenance_tracker.get_family_hierarchy()

# 保存到文件
provenance_tree_path = os.path.join(output_root, "relations", "provenance_tree.json")
os.makedirs(os.path.dirname(provenance_tree_path), exist_ok=True)

with open(provenance_tree_path, "w") as f:
    json.dump({
        "parent_map": family_result["parent_map"],
        "orphan_count": family_result["orphan_count"],
        "orphan_ids": family_result["orphan_ids"],
        "statistics": provenance_tracker.get_statistics(),
    }, f, indent=2)

console(f"✅ Provenance tree saved: {provenance_tree_path}")
console(f"📊 Statistics: {provenance_tracker.get_statistics()}")
console(f"⚠️ Orphans: {family_result['orphan_count']}")
```

---

### 第六步：驗證與測試

#### 6.1 測試單場景

```bash
# 使用測試配置
PYTHONPATH=src python -m my3dis.run_workflow \
  --config configs/multiscan/test_65.yaml
```

**檢查點：**
1. ✅ `filtered.json` 包含 `unique_id`, `parent_unique_id`, `ssam_frame_idx`
2. ✅ `provenance_tree.json` 生成成功
3. ✅ Orphan count 比之前的 `sam2_tree.json` 更低

#### 6.2 比較兩種 tree building 方法

創建比較腳本：`scripts/compare_trees.py`

```python
import json
import sys

def load_tree(path):
    with open(path) as f:
        return json.load(f)

provenance_tree = load_tree("provenance_tree.json")
containment_tree = load_tree("sam2_tree.json")

print("=== Tree Comparison ===")
print(f"Provenance-based families: {len([v for v in provenance_tree['parent_map'].values() if v is not None])}")
print(f"Containment-based families: {len(containment_tree.get('objects', {}))}")
print(f"Provenance orphans: {provenance_tree['orphan_count']}")
print(f"Containment orphans: {sum(1 for obj in containment_tree['objects'].values() if obj.get('parent') is None and obj.get('level') != 2)}")
```

---

## 實作順序建議

### **階段一：ID 修復與基礎架構（1-2 天）**

**目標**：修復 SSAM unique ID 衝突問題，建立 provenance tracking 基礎設施。

**任務清單**：
1. ✅ 修改 `semantic_refinement.py`：
   - 加入 `ssam_frame_idx` 參數
   - 使用 composite key 生成 unique_id
   - 記錄 `ssam_frame_idx` 和 `lineage`

2. ✅ 修改 `ssam_progressive_adapter.py`：
   - 傳遞 `f_idx` 給 `progressive_refinement_masks`

3. ✅ 驗證 `generate_candidates.py`：
   - 確認 metadata 正確保存到 `filtered.json`

4. ✅ 創建 `tracking/provenance_tracker.py`：
   - 實現 `ProvenanceTracker` class

**驗證標準**：
- ✅ 不同 frame 的 mask 有不同的 unique_id
- ✅ `filtered.json` 包含所有必要欄位

---

### **階段二：Dedup Rejection Tracking（2-3 天）**

**目標**：記錄 dedup rejection 時的 SSAM → SAM2 映射。

**任務清單**：
1. ✅ 修改 `stores.py`：
   - `filter_candidates` 返回 rejected candidates
   - 實現 `find_overlapping_mask_index`

2. ✅ 修改 `sam2_runner.py`：
   - 整合 `ProvenanceTracker`
   - 記錄 accepted/rejected prompts
   - 導出 `provenance_tree.json`

3. ✅ 測試單場景：
   - 運行 `test_65.yaml`
   - 檢查輸出文件

**驗證標準**：
- ✅ `provenance_tree.json` 生成成功
- ✅ Rejection rate > 0（表示有記錄到 dedup rejections）
- ✅ Orphan count 減少

---

### **階段三：比較與優化（1-2 天）**

**目標**：比較 provenance-based 和 containment-based tree，驗證改進效果。

**任務清單**：
1. ✅ 創建 `scripts/compare_trees.py`
2. ✅ 運行多場景實驗
3. ✅ 分析 orphan 減少幅度
4. ✅ 視覺化比較（可選）

**成功指標**：
- ✅ Provenance-based tree 的 family 數量 ≥ containment-based
- ✅ Orphan count 減少 30-50%
- ✅ False positive rate < 5%（手動抽查）

---

## 預期成果

### **數值預期**

假設原本的 containment-based tree：
- Total objects: 1000
- Families (有 parent 的 objects): 400
- Orphans: 600 (60%)

Provenance-based tree 預期：
- Total objects: 1000
- Families: 700-800
- Orphans: 200-300 (20-30%)

**改善幅度**：Orphan rate 從 60% 降到 20-30%，約 **50% 的改善**。

### **檔案輸出**

新增檔案：
```
outputs/experiments/<scene>/<run>/
├─ relations/
│  ├─ sam2_tree.json           # 原有的 containment-based tree
│  └─ provenance_tree.json     # ✅ 新增：provenance-based tree
└─ provenance_stats.json       # ✅ 新增：統計信息
```

---

## 潛在問題與解決方案

### **問題 1：Frame Mask Index 追蹤複雜度**

**挑戰**：需要追蹤 dedup_store 中每個 mask 的 index 對應到哪個 sam2_obj_id。

**當前方案**（簡單有效）：
- 使用外部字典 `dedup_mask_to_sam2_id: Dict[Tuple[int, int], int]`
- Key: (frame_idx, mask_index_in_store)
- Value: sam2_obj_id
- 在每次 accepted prompt 時記錄映射

**Alternative（可選優化）**：
- 在 `DedupStore` 內部加入 `mask_metadata: List[Dict]` 欄位
- 每次 `add_mask` 時同時記錄 metadata（包含 sam2_obj_id）
- 優點：封裝性更好
- 缺點：需要修改 DedupStore 內部實現

### **問題 2：Provenance 資訊不完整**

**場景**：某些舊實驗的 `filtered.json` 沒有 `unique_id` 欄位。

**解決方案**：
- 在 `ProvenanceTracker.register_accepted_prompt` 中檢查 `ssam_unique_id is None`
- 若為 None，則 fallback 到 containment-based 方法

### **問題 3：Rejection Rate 過高**

**場景**：如果 rejection rate > 80%，表示大部分 SSAM masks 都被 dedup 丟棄。

**原因分析**：
- SAM2 propagation 效果太好，SSAM 的新 masks 大多是重複的
- 或者 IoU threshold 設太低

**解決方案**：
- 這其實是好事！表示 SAM2 成功追蹤了大部分 objects
- 確保 rejection mapping 正確記錄，orphans 就會減少

---

## 總結

### ✅ 核心改進

1. **修復 unique ID 衝突**：使用 composite key 確保全局唯一
2. **記錄 dedup rejections**：捕捉被丟棄的 SSAM masks 對應的 SAM2 objects
3. **Provenance-based tree**：利用 SSAM 的原始親子關係，不依賴事後幾何分析

### 🎯 預期效果

- ✅ Orphan rate 從 60% 降到 20-30%
- ✅ Family 數量增加 50-100%
- ✅ 無需調整 containment threshold（避免 false positives）

### 📅 開發時程

- **Week 1**：ID 修復 + ProvenanceTracker（2-3 天）
- **Week 2**：Dedup rejection tracking + 整合（2-3 天）
- **Week 3**：測試 + 比較 + 文檔（2-3 天）

**Total: 6-9 天**

---

## 快速參考（Quick Reference）

### 核心改動總覽

| 模組 | 文件 | 改動類型 | 說明 |
|------|------|---------|------|
| **SSAM** | `semantic_refinement.py` | 🔧 修復 | 使用 composite key 生成唯一 ID |
| **SSAM** | `ssam_progressive_adapter.py` | ➕ 新增參數 | 傳遞 `ssam_frame_idx` |
| **Tracking** | `provenance_tracker.py` | ✨ 新文件 | 統一映射方案，追蹤 SSAM → SAM2 |
| **Tracking** | `stores.py` | 🔧 修改 | `filter_candidates` 返回 rejected info |
| **Tracking** | `sam2_runner.py` | ➕ 整合 | 使用 ProvenanceTracker 構建 tree |

### 關鍵數據結構

```python
# ProvenanceTracker 內部
ssam_to_sam2: Dict[str, int]
# 範例：{'0050_2_0001': 1, '0100_2_0001': 1, '0100_4_0002': 3}

sam2_provenance: Dict[int, Dict]
# 範例：{
#   1: {'ssam_unique_id': '0050_2_0001', 'ssam_parent_id': None, ...},
#   3: {'ssam_unique_id': '0100_4_0002', 'ssam_parent_id': '0100_2_0001', ...}
# }
```

### 預期輸出格式

**Unique ID 格式**：`{frame:04d}_{level}_{seq:04d}`
- 範例：`0050_2_0001`（Frame 50, Level 2, Sequence 1）

**Provenance Tree JSON**：
```json
{
  "parent_map": {
    "1": null,
    "2": 1,
    "3": 1
  },
  "orphan_count": 0,
  "orphan_ids": [],
  "statistics": {
    "total_ssam_masks": 150,
    "accepted_prompts": 50,
    "rejected_prompts": 100,
    "rejection_rate": 0.67
  }
}
```

### 驗證檢查清單

**階段一驗證**：
- [ ] `filtered.json` 包含 `unique_id`, `parent_unique_id`, `ssam_frame_idx`
- [ ] Unique ID 格式正確：`\d{4}_\d_\d{4}`
- [ ] 不同 frame 的 mask 有不同的 unique_id

**階段二驗證**：
- [ ] `provenance_tree.json` 生成成功
- [ ] `rejection_rate > 0`（表示有記錄到 dedup rejections）
- [ ] `orphan_count < 原本的 containment tree 的 orphan count`

**階段三驗證**：
- [ ] Provenance tree 的 family 數量 ≥ containment tree
- [ ] Orphan count 減少 30-50%
- [ ] 手動抽查 10 個 parent-child 關係，確認準確性
