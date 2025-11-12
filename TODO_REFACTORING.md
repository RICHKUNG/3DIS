# My3DIS 重構 TODO 清單

> **最後更新**: 2025-11-07 晚上
> **當前階段**: Phase A 收尾

---

## 🔥 立即待辦 (完成 Phase A)

### [ ] 1. 啟用新版 scene_workflow.py

**預估時間**: 10 分鐘

```bash
cd /media/Pluto/richkung/My3DIS/src/my3dis/workflow

# 備份原檔案
mv scene_workflow.py scene_workflow_legacy.py

# 啟用新版本
mv scene_workflow_v2.py scene_workflow.py
```

**驗證步驟**:
```bash
# 確認可以 import
PYTHONPATH=src python3 -c "from my3dis.workflow.scene_workflow import SceneWorkflow; print('✓ Import successful')"
```

---

### [ ] 2. 測試新版 workflow

**預估時間**: 20 分鐘

**測試項目**:
- [ ] 測試 SSAM stage 獨立運行
- [ ] 測試 Filter stage
- [ ] 測試完整 workflow (如果有測試配置)
- [ ] 確認所有 stage runners 正常執行

**測試腳本** (需要創建):
```bash
# 測試基本功能
PYTHONPATH=src python3 scripts/test_scene_workflow_v2.py
```

---

### [ ] 3. 更新文檔

**預估時間**: 10 分鐘

**需要更新的文檔**:
- [ ] `REFACTORING_PROGRESS_2025_11_07.md` - 標記 Phase A 完成
- [ ] `CLAUDE.md` - 添加 StageRunner 使用範例
- [ ] `docs/ARCHITECTURE.md` - 更新架構圖

**範例內容** (添加到 CLAUDE.md):
```markdown
### Stage Runner Pattern (NEW - 2025-11-07)

The workflow now uses a StageRunner pattern for better modularity:

```python
from my3dis.workflow.stage_runner import StageRunner

class CustomStageRunner(StageRunner):
    @property
    def name(self) -> str:
        return "custom"

    def should_run(self) -> bool:
        return self._stage_cfg().get('enabled', True)

    def execute(self) -> None:
        # Custom stage logic
        print(f"Executing {self.name} stage...")
\```
```

---

### [ ] 4. 清理與提交

**預估時間**: 5 分鐘

- [ ] 刪除不需要的測試檔案
- [ ] 確認所有新檔案已加入 git
- [ ] (可選) 創建 git commit

```bash
# 檢查新增檔案
git status

# 查看改動
git diff src/my3dis/workflow/scene_workflow.py
```

---

## ⏳ 後續工作 (Phase B & C)

### Phase 3: CLI 分離 (未開始)

**預估時間**: 2-4 小時

**目標**:
- [ ] 建立 `src/my3dis/services/` 目錄
- [ ] 從 `generate_candidates.py` 提取業務邏輯到 `services/candidate_generation.py`
- [ ] 從 `filter_candidates.py` 提取業務邏輯到 `services/candidate_filtering.py`
- [ ] 從 `track_from_candidates.py` 提取業務邏輯到 `services/tracking.py`
- [ ] 簡化 CLI 檔案至 < 100 行 (僅 argparse + 調用)
- [ ] 測試業務邏輯可以獨立運行

**優先順序**: 中
**前置依賴**: Phase A 完成

---

### Phase 4: Pydantic Config Models (未開始)

**預估時間**: 3-4 小時

**目標**:
- [ ] 安裝 pydantic (如果尚未安裝)
- [ ] 建立 `src/my3dis/config/models.py`
- [ ] 創建 `WorkflowConfigModel`
- [ ] 創建 `SSAMStageModel`
- [ ] 創建 `TrackerStageModel`
- [ ] 創建其他 stage models
- [ ] 整合到 `SceneWorkflow.__init__()`
- [ ] 測試配置驗證

**優先順序**: 中-低
**前置依賴**: 無 (可獨立進行)

---

## ✅ 已完成工作

### Phase C: 整合與測試 ✅
- [x] 創建測試套件 `scripts/test_refactoring_phase1.py`
- [x] 測試所有新模組 (8/8 通過)
- [x] 驗證向後相容性
- [x] 更新 CLAUDE.md 文檔

### Phase 1: common_utils.py 模組化 ✅
- [x] 建立 `src/my3dis/io/` 目錄
- [x] 建立 `src/my3dis/mask/` 目錄
- [x] 建立 `src/my3dis/utils/` 目錄
- [x] 創建 7 個專門模組
- [x] 將 common_utils.py 改為相容層 (306 → 109 行)
- [x] 測試驗證

### Phase A: scene_workflow.py 拆分 (90%)
- [x] 創建 StageRunner 抽象介面
- [x] 創建 6 個 stage runner 實作
- [x] 創建新版 scene_workflow_v2.py (664 → 295 行)
- [ ] 啟用新版本 ⬅️ **當前位置**
- [ ] 測試驗證
- [ ] 文檔更新

---

## 📊 進度追蹤

| Phase | 狀態 | 進度 | 預估剩餘時間 |
|-------|------|------|-------------|
| **Phase C** | ✅ 完成 | 100% | - |
| **Phase 1** | ✅ 完成 | 100% | - |
| **Phase A** | 🔄 進行中 | 90% | **40 分鐘** |
| **Phase 3** | ⏳ 未開始 | 0% | 2-4 小時 |
| **Phase 4** | ⏳ 未開始 | 0% | 3-4 小時 |
| **總計** | 🔄 | **63%** | **~6 小時** |

---

## 📝 備註

### 已知問題
1. **scene_workflow_v2.py 未測試**
   - 新版本已創建但未實際運行測試
   - 需要確認所有 stage runners 正常工作

2. **Context 狀態管理**
   - `_run_dir` 和 `_manifest` 通過 context 傳遞
   - 可能需要改為更明確的模式

### 技術債
1. **測試覆蓋率仍低**
   - Phase 1 有測試，但 Phase A 缺少 integration tests
   - 建議添加 end-to-end 測試

2. **文檔更新滯後**
   - ARCHITECTURE.md 需要更新架構圖
   - 需要添加 StageRunner 使用教學

---

## 🎯 本週目標

**主要目標**: 完成 Phase A

- [ ] 星期四晚: 啟用新版 scene_workflow ✅ (當前)
- [ ] 星期五早: 測試驗證
- [ ] 星期五午: 文檔更新
- [ ] 星期五晚: Phase A 完成總結

**延伸目標** (可選):
- [ ] 開始 Phase 3 規劃
- [ ] 研究 Pydantic 整合方案

---

**檔案位置**: `/media/Pluto/richkung/My3DIS/TODO_REFACTORING.md`
**相關文檔**:
- `REFACTORING_STATUS_2025_11_07.md` - 詳細狀態報告
- `REFACTORING_PROGRESS_2025_11_07.md` - 進度追蹤
- `docs/REFACTORING_PLAN_2025_11.md` - 重構計畫
