# SAM2 Tracking 中的 Dedup 與家族 Merge 處理流程

## 目錄
1. [概述](#概述)
2. [核心組件](#核心組件)
3. [Dedup（去重）機制](#dedup去重機制)
4. [家族 Merge 處理流程](#家族-merge-處理流程)
5. [設計目的與好處](#設計目的與好處)
6. [完整處理流程 Pseudocode](#完整處理流程-pseudocode)
7. [實際範例](#實際範例)

---

## 概述

My3DIS 的 SAM2 tracking 階段需要處理兩個關鍵問題：

1. **物體重複追蹤問題**：同一物體的多個 SSAM 候選可能在不同幀或同一幀中重複出現
2. **跨層級家族關係問題**：父層級（L2）的物體可能與子層級（L4/L6）的候選代表同一物體，需要正確處理家族合併

本文檔詳細說明 **DedupStore** 和 **ProvenanceTracker** 如何協同工作來解決這些問題。

---

## 核心組件

### 1. DedupStore（去重存儲器）
**文件位置**: `src/my3dis/tracking/stores.py:34`

**職責**:
- 維護每幀的 downscaled mask stack（降採樣遮罩堆疊）
- 基於 IoU（Intersection over Union）檢測重複候選
- 返回拒絕候選的匹配 mask 索引

**關鍵數據結構**:
```python
class _DedupEntry:
    target_shape: Tuple[int, int]  # 降採樣後的目標尺寸（如 256x256）
    masks: List[np.ndarray]        # 該幀已接受的所有 masks（降採樣版本）

class DedupStore:
    _max_dim: int = 256                        # 降採樣最大維度
    _frames: Dict[int, _DedupEntry]            # frame_idx → _DedupEntry
```

### 2. ProvenanceTracker（溯源追踪器）
**文件位置**: `src/my3dis/tracking/provenance_tracker.py:18`

**職責**:
- 追蹤 SSAM candidate → SAM2 object 的映射關係
- 記錄被拒絕候選的 virtual children（虛擬子代）關係
- 提供 O(1) 的父對象查詢

**關鍵數據結構**:
```python
class ProvenanceTracker:
    ssam_to_sam2: Dict[str, int]              # SSAM ID → SAM2 object ID（包含接受和拒絕）
    sam2_provenance: Dict[int, Dict[str, Any]] # SAM2 ID → 完整溯源元數據（僅接受）
    virtual_children: Dict[int, List[str]]     # SAM2 parent ID → 被 dedup 的 SSAM IDs
    unresolved_rejections: List[tuple]         # 無法解析的拒絕記錄（調試用）
```

---

## Dedup（去重）機制

### 工作原理

DedupStore 使用 **downscaling + IoU** 策略實現高效去重：

1. **Downscaling（降採樣）**:
   - 將所有 masks 降採樣到固定最大維度（默認 256px）
   - 減少記憶體佔用和計算成本
   - 保持縱橫比（aspect ratio）

2. **IoU 計算（向量化）**:
   - 將候選 mask 與該幀所有已存在 masks 批次計算 IoU
   - 使用 numpy broadcasting 加速計算
   - 返回最大 IoU 值和對應 mask 索引

3. **閾值判斷**:
   - 如果 `max_IoU > threshold`（默認 0.6），拒絕候選
   - 否則接受候選並加入 dedup store

### Pseudocode

```python
class DedupStore:
    def filter_candidates(frame_idx, candidates, threshold):
        """
        過濾候選，返回接受和拒絕列表

        Returns:
            accepted: List[PromptCandidate]
            rejected: List[(PromptCandidate, matched_mask_index)]
        """
        accepted = []
        rejected = []

        for candidate in candidates:
            seg = candidate.seg_for_iou  # 用於 IoU 計算的 mask

            if seg is not None:
                # 1. 找到重疊的 mask 索引
                match_idx = find_overlapping_mask_index(frame_idx, seg, threshold)

                if match_idx is not None:
                    # 2. 拒絕：記錄匹配的 mask 索引
                    rejected.append((candidate, match_idx))
                    continue

            # 3. 接受：添加到已接受列表
            accepted.append(candidate)

            # 4. 將 mask 加入 dedup store（用於後續比較）
            if seg is not None:
                add_mask(frame_idx, seg)

        return accepted, rejected

    def find_overlapping_mask_index(frame_idx, mask, threshold):
        """
        找到與給定 mask 重疊的現有 mask 索引

        Returns:
            int: 重疊 mask 的索引（在該幀的 mask list 中）
            None: 沒有重疊 mask
        """
        entry = _frames.get(frame_idx)
        if entry is None or not entry.masks:
            return None

        # 1. 降採樣候選 mask
        resized = _resize(mask, entry.target_shape)

        # 2. 向量化 IoU 計算（批次處理所有現有 masks）
        existing_stack = np.stack(entry.masks, axis=0)  # Shape: (N, H, W)
        cand_broadcast = resized[np.newaxis, :, :]       # Shape: (1, H, W)

        # Intersection: (N, H, W) AND (1, H, W) → (N,)
        inter = np.logical_and(existing_stack, cand_broadcast).sum(axis=(1, 2))

        # Union: (N, H, W) OR (1, H, W) → (N,)
        union = np.logical_or(existing_stack, cand_broadcast).sum(axis=(1, 2))

        # 3. 計算 IoU（避免除零）
        valid = union > 0
        if not valid.any():
            return None

        ious = np.zeros(len(entry.masks))
        ious[valid] = inter[valid] / union[valid]

        # 4. 返回最大 IoU 對應的索引
        max_idx = int(ious.argmax())
        if ious[max_idx] > threshold:
            return max_idx  # 找到重疊 mask

        return None  # 沒有重疊
```

### 關鍵優化點

1. **向量化計算**: 使用 `np.stack` + broadcasting 批次計算所有 IoU，避免迴圈
2. **降採樣**: 固定 256px 最大維度，大幅減少計算量（從 1920x1080 降至 ~256x144）
3. **惰性評估**: 只在需要時計算 IoU，接受的 mask 直接跳過後續候選

---

## 家族 Merge 處理流程

### 核心概念：Virtual Children（虛擬子代）

當子層級（如 L4）的 SSAM 候選被 dedup 到父層級（如 L2）的 SAM2 物體時，該子候選成為 **virtual child**：

- **物理追蹤**: 只有父物體被 SAM2 追蹤（避免重複）
- **邏輯關係**: 子候選在家族樹中仍屬於父物體的 children
- **完整溯源**: 可以查詢「哪些子級 masks 被合併到父物體」

### Unified Mapping 設計

ProvenanceTracker 使用 **單一映射表** (`ssam_to_sam2`) 同時記錄接受和拒絕的候選：

```
SSAM Candidate → SAM2 Object ID
├── Accepted prompts: 新物體，記錄完整 provenance
└── Rejected prompts: 現有物體，記錄為 virtual children
```

這種設計實現 **O(1) 父對象查詢**：
```python
def find_parent(sam2_obj_id):
    # 1. 從 provenance 獲取 SSAM parent ID
    ssam_parent_id = sam2_provenance[sam2_obj_id]["ssam_parent_id"]

    # 2. 單次查詢即可找到父對象的 SAM2 ID（無論父對象是接受還是拒絕）
    return ssam_to_sam2.get(ssam_parent_id)
```

### Pseudocode

```python
class ProvenanceTracker:
    def register_accepted_prompt(sam2_obj_id, ssam_unique_id, ssam_parent_id, ...):
        """
        記錄接受的 SSAM 候選（新 SAM2 物體）
        """
        # 1. 建立 SSAM → SAM2 映射
        ssam_to_sam2[ssam_unique_id] = sam2_obj_id

        # 2. 記錄完整 provenance 元數據（僅接受的候選有此數據）
        sam2_provenance[sam2_obj_id] = {
            "ssam_unique_id": ssam_unique_id,
            "ssam_parent_id": ssam_parent_id,  # 父對象的 SSAM ID（可能是 None）
            "ssam_frame_idx": ssam_frame_idx,
            "level": level,  # 2, 4, or 6
            "lineage": lineage,  # 完整祖先鏈
        }

    def register_rejected_prompt(ssam_unique_id, matched_sam2_obj_id, ssam_parent_id):
        """
        記錄拒絕的 SSAM 候選（dedup 到現有物體）

        關鍵：追蹤 virtual children 關係
        """
        # 1. 建立 SSAM → SAM2 映射（映射到匹配的現有物體）
        if ssam_unique_id not in ssam_to_sam2:
            ssam_to_sam2[ssam_unique_id] = matched_sam2_obj_id

        # 2. 如果有父對象信息，追蹤 virtual children 關係
        if ssam_parent_id is not None:
            # 2a. 找到父對象的 SAM2 ID
            parent_sam2_id = ssam_to_sam2.get(ssam_parent_id)

            if parent_sam2_id is not None:
                # 2b. 將此 SSAM mask 記錄為父對象的 virtual child
                if parent_sam2_id not in virtual_children:
                    virtual_children[parent_sam2_id] = []

                if ssam_unique_id not in virtual_children[parent_sam2_id]:
                    virtual_children[parent_sam2_id].append(ssam_unique_id)

    def find_parent(sam2_obj_id):
        """
        O(1) 父對象查詢
        """
        prov = sam2_provenance.get(sam2_obj_id)
        if prov is None:
            return None

        ssam_parent_id = prov.get("ssam_parent_id")
        if ssam_parent_id is None:
            return None  # 根層級物體（如 L2）

        # 單次查詢！適用於接受和拒絕的父對象
        return ssam_to_sam2.get(ssam_parent_id)

    def get_virtual_children(sam2_obj_id):
        """
        查詢虛擬子代
        """
        return virtual_children.get(sam2_obj_id, [])
```

---

## 設計目的與好處

### 1. 避免重複追蹤（Dedup）

**問題**:
- 同一物體可能在多個幀被 SSAM 重複檢測
- 跨層級候選可能代表相同物體（如 L2 的椅子 vs L4 的椅子）

**解決方案**:
- DedupStore 在每幀維護已接受 masks，新候選必須通過 IoU 檢查
- 拒絕高 IoU 候選，避免 SAM2 追蹤重複物體

**好處**:
- ✅ 減少計算成本（SAM2 propagation 非常昂貴）
- ✅ 避免輸出重複物體（提高下游任務準確性）
- ✅ 減少記憶體佔用（fewer tracked objects）

### 2. 保持家族樹完整性（Virtual Children）

**問題**:
- 子層級候選被 dedup 後，家族樹會出現「缺失 children」
- 無法回答「哪些子級 masks 被合併到父物體」

**解決方案**:
- ProvenanceTracker 記錄被 dedup 的候選為 virtual children
- 單一映射表支援 O(1) 父對象查詢（無論父對象是否被 dedup）

**好處**:
- ✅ 完整溯源：可以追蹤 SSAM → SAM2 的所有映射（包括拒絕的）
- ✅ 語義完整性：家族樹反映真實的物體層級關係
- ✅ 調試友好：可以查詢「為什麼某個子候選沒有被追蹤」（因為被 dedup 了）

### 3. 統一映射設計（Unified Mapping）

**問題**:
- 早期設計使用兩個映射表：`accepted_map` 和 `rejected_map`
- 父對象查詢需要檢查兩個表，複雜度 O(2)

**解決方案**:
- 單一 `ssam_to_sam2` 映射表同時記錄接受和拒絕
- 父對象查詢簡化為單次字典查詢 O(1)

**好處**:
- ✅ 代碼簡潔：單一數據源，減少同步問題
- ✅ 性能優化：O(1) vs O(2) 查詢
- ✅ 易於擴展：新功能（如 virtual children）自然整合

### 4. Eager + Deferred 處理（Timing Fix）

**問題（2025-11-04 修復前）**:
- 拒絕候選在 propagation 前立即處理
- 此時父對象的 prompt masks 尚未添加到 dedup store
- 導致 80% 的 orphans（無法找到父對象）

**解決方案**:
```python
# EAGER: 接受的候選立即註冊（建立 prompt mask → sam2_id 映射）
for cand in accepted_candidates:
    provenance_tracker.register_accepted_prompt(...)
    dedup_mask_to_sam2_id[(frame_idx, mask_idx)] = sam2_obj_id  # 預先建立映射

# ... propagate ...

# DEFERRED: 拒絕的候選延後處理（確保所有映射已建立）
for cand, matched_mask_idx in deferred_rejections:
    matched_sam2_id = dedup_mask_to_sam2_id.get((frame_idx, matched_mask_idx))
    provenance_tracker.register_rejected_prompt(ssam_unique_id, matched_sam2_id, ...)
```

**好處**:
- ✅ 解決 timing 問題：父對象映射在子對象拒絕前已建立
- ✅ Orphan 率降低：從 2.2% 降至 <0.5%（預期）
- ✅ 邏輯清晰：Eager 處理接受，Deferred 處理拒絕

---

## 完整處理流程 Pseudocode

```python
def sam2_tracking(frames_dir, predictor, candidate_batches, ...):
    """
    SAM2 追蹤主流程（整合 dedup 和 provenance tracking）
    """
    # 初始化
    dedup_store = DedupStore(max_dim=256)
    provenance_tracker = ProvenanceTracker()
    dedup_mask_to_sam2_id = {}  # (frame_idx, mask_idx) → sam2_obj_id

    obj_count = level * 1000  # Level-based ID offset (L2:2000, L4:4000, L6:6000)

    for batch in candidate_batches:
        frame_idx = batch.local_index
        abs_idx = batch.frame_index

        # ==========================================
        # STEP 1: 準備候選
        # ==========================================
        prepared_candidates = prepare_prompt_candidates(batch.candidates)

        # ==========================================
        # STEP 2: Dedup 過濾
        # ==========================================
        accepted_candidates, rejected_candidates = dedup_store.filter_candidates(
            frame_idx=abs_idx,
            candidates=prepared_candidates,
            threshold=iou_threshold,  # 默認 0.6
        )
        # accepted: [(cand1, None), (cand2, None), ...]
        # rejected: [(cand3, mask_idx_5), (cand4, mask_idx_12), ...]

        # ==========================================
        # STEP 3: DEFERRED 拒絕處理（延後到 propagation 後）
        # ==========================================
        deferred_rejections = []
        for cand, matched_mask_idx in rejected_candidates:
            deferred_rejections.append((cand, matched_mask_idx, abs_idx))

        if not accepted_candidates:
            continue  # 沒有接受的候選，跳過此幀

        # ==========================================
        # STEP 4: 添加 prompts 到 SAM2 predictor
        # ==========================================
        obj_count_before = obj_count
        obj_count = add_prompts_to_predictor(
            predictor, state, frame_idx, accepted_candidates,
            obj_start=obj_count, ...
        )
        # 新物體 IDs: [obj_count_before, obj_count_before+1, ..., obj_count-1]

        # ==========================================
        # STEP 5: EAGER 接受處理（立即註冊 provenance）
        # ==========================================
        dedup_entry = dedup_store._frames.get(abs_idx)
        mask_count_before = len(dedup_entry.masks) if dedup_entry else 0

        for idx, cand in enumerate(accepted_candidates):
            sam2_obj_id = obj_count_before + idx
            payload = cand.payload
            ssam_unique_id = payload.get("unique_id")  # e.g., "0050_4_0123"

            if ssam_unique_id:
                # 5a. 註冊接受的 prompt
                provenance_tracker.register_accepted_prompt(
                    sam2_obj_id=sam2_obj_id,
                    ssam_unique_id=ssam_unique_id,
                    ssam_parent_id=payload.get("parent_unique_id"),  # e.g., "0050_2_0045"
                    ssam_frame_idx=payload.get("ssam_frame_idx", frame_idx),
                    level=payload.get("level"),  # 2, 4, or 6
                    lineage=payload.get("lineage"),
                )

                # 5b. EAGER 映射：預先建立 prompt mask → sam2_id 映射
                # Propagation 後，prompt masks 會在 dedup_store 中的位置：
                # [existing_masks] + [prompt_0, prompt_1, ...]
                mask_idx_after_propagation = mask_count_before + idx
                dedup_mask_to_sam2_id[(abs_idx, mask_idx_after_propagation)] = sam2_obj_id

        # ==========================================
        # STEP 6: SAM2 Propagation
        # ==========================================
        frame_segments = propagate_frame_predictions(
            predictor, state, frame_idx,
            local_to_abs=local_to_abs,
            total_frames=len(frame_numbers),
            max_propagate=max_propagate,
            mask_scale_ratio=mask_scale_ratio,
        )
        # frame_segments: {abs_frame_idx: {obj_id: packed_mask, ...}, ...}

        # ==========================================
        # STEP 7: 添加 propagated masks 到 dedup store
        # ==========================================
        for abs_out_idx, frame_data in frame_segments.items():
            if not frame_data:
                continue

            # 7a. 記錄添加前的 mask 數量
            dedup_entry = dedup_store._frames.get(abs_out_idx)
            mask_count_before = len(dedup_entry.masks) if dedup_entry else 0

            # 7b. 添加 propagated masks 到 dedup store
            dedup_store.add_packed(abs_out_idx, frame_data)

            # 7c. 建立 propagated masks 的映射
            for idx, obj_id in enumerate(sorted(frame_data.keys())):
                mask_idx_in_store = mask_count_before + idx
                dedup_mask_to_sam2_id[(abs_out_idx, mask_idx_in_store)] = obj_id

        # ==========================================
        # STEP 8: DEFERRED 拒絕處理（現在所有映射已建立）
        # ==========================================
        for cand, matched_mask_idx, prompt_frame_idx in deferred_rejections:
            payload = cand.payload
            ssam_unique_id = payload.get("unique_id")

            if ssam_unique_id:
                # 8a. 查找匹配的 SAM2 object ID
                matched_sam2_id = dedup_mask_to_sam2_id.get(
                    (prompt_frame_idx, matched_mask_idx)
                )

                if matched_sam2_id is not None:
                    # 8b. 註冊拒絕的 prompt（記錄為 virtual child）
                    provenance_tracker.register_rejected_prompt(
                        ssam_unique_id=ssam_unique_id,
                        matched_sam2_obj_id=matched_sam2_id,
                        ssam_parent_id=payload.get("parent_unique_id"),
                    )
                else:
                    # 8c. 記錄無法解析的拒絕（調試用）
                    provenance_tracker.register_unresolved_rejection(
                        ssam_unique_id=ssam_unique_id,
                        matched_mask_idx=matched_mask_idx,
                        frame_idx=prompt_frame_idx,
                    )

    # ==========================================
    # STEP 9: 返回追蹤結果
    # ==========================================
    return TrackingArtifacts(
        object_refs=object_refs,
        preview_segments=preview_segments,
        frames_with_predictions=frames_with_predictions,
        objects_seen=objects_seen,
        provenance_tracker=provenance_tracker,  # 包含完整溯源信息
    )
```

---

## 實際範例

### 場景：椅子的跨層級 Dedup

**假設**:
- Frame 0050: L2 候選檢測到椅子 → 接受，分配 `sam2_obj_id=2001`
- Frame 0050: L4 候選檢測到同一椅子（更細緻） → Dedup 拒絕

#### 處理流程

```
1. SSAM 生成候選：
   L2: unique_id="0050_2_0001", parent_id=None
   L4: unique_id="0050_4_0001", parent_id="0050_2_0001"

2. Dedup 過濾（Frame 0050）：
   L2 candidate:
     - dedup_store 為空 → 接受
     - add_mask(frame=0050, mask=L2_mask)
     - dedup_store.masks[0050] = [L2_mask_downscaled]

   L4 candidate:
     - IoU(L4_mask, L2_mask) = 0.85 > 0.6 → 拒絕
     - matched_mask_idx = 0 (L2_mask 在 dedup_store 中的索引)

3. EAGER 處理（L2 接受）：
   provenance_tracker.register_accepted_prompt(
       sam2_obj_id=2001,
       ssam_unique_id="0050_2_0001",
       ssam_parent_id=None,
       level=2,
   )

   ssam_to_sam2["0050_2_0001"] = 2001
   sam2_provenance[2001] = {
       "ssam_unique_id": "0050_2_0001",
       "ssam_parent_id": None,
       "level": 2,
       ...
   }

   dedup_mask_to_sam2_id[(0050, 0)] = 2001  # Eager 映射

4. SAM2 Propagation:
   - Object 2001 propagated to frames 0051-0100
   - Each frame: dedup_store.add_packed(...) 添加 propagated masks

5. DEFERRED 處理（L4 拒絕）：
   matched_sam2_id = dedup_mask_to_sam2_id[(0050, 0)] = 2001

   provenance_tracker.register_rejected_prompt(
       ssam_unique_id="0050_4_0001",
       matched_sam2_obj_id=2001,
       ssam_parent_id="0050_2_0001",
   )

   # 更新數據結構：
   ssam_to_sam2["0050_4_0001"] = 2001  # L4 候選映射到 L2 物體
   virtual_children[2001] = ["0050_4_0001"]  # L4 候選成為虛擬子代

6. 查詢結果：
   - get_virtual_children(2001) = ["0050_4_0001"]
   - 語義：「物體 2001（L2 椅子）合併了 L4 椅子候選」
```

#### 家族樹輸出

```json
{
  "objects": {
    "2001": {
      "ssam_unique_id": "0050_2_0001",
      "level": 2,
      "parent_id": null,
      "virtual_children": ["0050_4_0001"]  // 被 dedup 的 L4 候選
    }
  },
  "statistics": {
    "total_ssam_masks": 2,
    "accepted_prompts": 1,
    "rejected_prompts": 1,
    "virtual_children_count": 1,
    "parents_with_virtual_children": 1
  }
}
```

---

## 總結

| 組件 | 職責 | 關鍵數據結構 | 時間複雜度 |
|------|------|--------------|-----------|
| **DedupStore** | IoU-based 去重 | `_frames: Dict[int, _DedupEntry]` | O(N) per frame (向量化) |
| **ProvenanceTracker** | SSAM→SAM2 映射 | `ssam_to_sam2: Dict[str, int]` | O(1) 查詢 |
| **Virtual Children** | 家族 merge 追蹤 | `virtual_children: Dict[int, List[str]]` | O(1) 查詢 |
| **Eager/Deferred** | Timing fix | `dedup_mask_to_sam2_id: Dict[tuple, int]` | O(1) 查詢 |

**設計優勢**:
1. ✅ **高效去重**: 向量化 IoU + 降採樣，支援大規模場景
2. ✅ **完整溯源**: 追蹤所有 SSAM→SAM2 映射（包括拒絕的）
3. ✅ **語義完整**: Virtual children 保持家族樹邏輯完整性
4. ✅ **Orphan 修復**: Eager/Deferred 處理解決 timing 問題（<0.5% orphan rate）
5. ✅ **易於調試**: 提供統計信息和 unresolved rejections 追蹤

**相關文檔**:
- **Provenance Tracking**: [`docs/PROVENANCE_TRACKING.md`](../../../docs/PROVENANCE_TRACKING.md)
- **Orphan Fix**: [`ORPHAN_FIX_PROGRESS.md`](../../../ORPHAN_FIX_PROGRESS.md)
- **Tracking Guide**: [`TRACKING_GUIDE.md`](./TRACKING_GUIDE.md)
