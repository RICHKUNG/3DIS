# My3DIS 新人指南

> 這份文件協助新進夥伴快速理解專案目的、環境設定與常用操作流程。請依自己的工作範疇挑選章節細讀。

## 執行流程圖（一眼看懂 Pipeline）

```
                  ┌─────────── 選擇模式 ───────────┐
                  │  1) YAML Orchestrator  2) 兩階段CLI │
                  └───────────────────────────────────┘
                            │                 │
             python -m my3dis.run_workflow    │
                            │                 │
         ▼ SceneWorkflow（單場景一輪）        │      兩階段 CLI：
         SSAM → Filter → Tracker → Report      │      1) 生成候選：
           │        │        │      │         │         my3dis.generate_candidates: run_generation
  generate_candidates  filter_candidates  track_from_candidates  generate_report
           │                     │                     │
  ssam_progressive_adapter       │          prepare_tracking_context
           │                     │          ensure_subset_video
  progressive_refinement.masks   │          level_runner.run_level_tracking × N level
           │                     │              ├─ candidate_loader.iter_candidate_batches
           └─ raw_archive/filtered           ├─ sam2_runner.sam2_tracking
                                              └─ outputs.(video|object|viz)
```

## 1. 專案核心概念
- 目標：以 **Semantic-SAM** 產生多層級遮罩候選，再透過 **SAM2** 進行追蹤補洞，最終輸出完整遮罩與報告。
- 主要操作：挑選場景與影格 → 多層 Semantic-SAM → 過濾候選 → SAM2 追蹤與補洞 → 匯出結果、紀錄執行資訊。
- 主要入口：`run_experiment.sh` 與 `python -m my3dis.run_workflow --config <YAML>`.
- 重要依賴：兩個 Conda 環境（`Semantic-SAM`、`SAM2`）、多重資料來源與 checkpoint。

## 2. 基礎環境與資源
- **資料集**：`./data/multiscan/<scene>/outputs/color`（唯讀，不要自行寫入）。
- **外部 Repo**：
  - Semantic-SAM → `../Semantic-SAM`
  - SAM2 → `../SAM2`
- **Checkpoint 預設路徑**：
  - Semantic-SAM SwinL：`../Semantic-SAM/checkpoints/swinl_only_sam_many2many.pth`
  - SAM2 YAML：`../SAM2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml`
  - SAM2 Weights：`../SAM2/checkpoints/sam2.1_hiera_large.pt`
- **常用環境變數**（必要時覆寫路徑）：
  - `MY3DIS_SEMANTIC_SAM_ROOT`, `MY3DIS_SEMANTIC_SAM_CKPT`
  - `MY3DIS_SAM2_ROOT`, `MY3DIS_SAM2_CFG`, `MY3DIS_SAM2_CKPT`
  - `MY3DIS_DATASET_ROOT`, `MY3DIS_DATA_PATH`, `MY3DIS_OUTPUT_ROOT`
- **專案環境快照**：`env/`（目前設定）、`archive/env/`（歷史紀錄）。

## 3. 目錄導覽
- `src/my3dis/`：核心程式碼（管線、追蹤、工具）。
- `config.py` (WIP)：嘗試中的型別化配置結構，未正式接到 workflow；目前 YAML 仍以 dict 形式傳遞
  - `semantic_refinement.py` ⭐：漸進式細化核心演算法與工具
  - `semantic_refinement_cli.py` ⭐：新的 CLI 入口
  - `progressive_refinement.py` ⚠️：僅供舊版匯入，會顯示棄用警告
- `configs/`：YAML 設定。`configs/multiscan/` 為 MultiScan 相關範本，`configs/index/` 提供場景索引。
- `scripts/`：批次執行、資料前處理腳本。
- `run_experiment.sh` / `run_experiment_sweep.sh`：預設工作流程的 Shell 入口。
- `run_workflow.py`：YAML 駆動的主程式，可靈活指定場景與參數。
- `logs/`：執行紀錄（工作日誌、PID mapping、workflow history）。
- `outputs/`：實驗輸出（依 timestamp 與參數自動分層）。
- `data/examples/`：示例資料與測試輸入。

## 4. 設定檔重點
- **主設定**：`configs/multiscan/base.yaml`
  1. `experiment`: 指定實驗名稱、資料位置、欲處理場景與輸出根目錄。
  2. `stages`: 控制各階段啟用與參數（SSAM、Filter、Tracker、Report）。
  3. `gpu` 欄位：設定使用 GPU index，或留 `null` 由程式判斷。
- **配置管理**：YAML 仍由 `src/my3dis/workflow/io.py` 讀成 dict；`src/my3dis/config.py` 的 `PipelineConfig` 是正在驗證的型別化方案，可在 notebook/測試中重複使用同一組驗證邏輯。
- **場景索引**：`configs/index/multiscan_scene_index.json` 列出所有場景、影格數量與對應 YAML。
- **覆寫技巧**：
  - 使用 YAML `experiment.tag` 幫助識別輸出資料夾。
  - 若只跑單一場景，填入 `experiment.scenes: [scene_xxx]` 或透過 `scene_start`/`scene_end` 控制範圍。
  - `experiment.frames.step` 決定 SAM2 會輸出的影格頻率；`ssam_freq` 僅影響 Semantic-SAM 呼叫頻率。
- **快速檢查**：修改後使用 `python -m my3dis.run_workflow --config <path> --dry-run`（若腳本提供）確認組態。

## 5. Pipeline 操作流程
1. **準備階段**
   - 確認 Conda 環境可用：`conda env list | grep Semantic-SAM`、`SAM2`.
   - 檢查資料與輸出磁碟空間，避免長時間任務失敗。
   - 視需求設定環境變數或調整 YAML。
2. **Semantic-SAM 候選產生**
   - 依 `levels`（例如 2、4、6）進行多層細化並記錄 `manifest.json`。
   - 每個 `level_{L}` 下會產生 `raw/manifest.json` 與 `chunk_*.tar.gz`（壓縮的 meta + mask NPZ），filter stage 可直接引用。
   - `ssam_freq` 控制 Semantic-SAM 呼叫頻率，`downscale_masks` / `mask_scale_ratio` 可先降解析度加速後續階段。
3. **候選過濾（Filter Stage）**
   - 以 `min_area`、`stability_threshold` 過濾低品質遮罩，輸出 `filtered/filtered.json`（已壓縮遮罩）。
   - 若 `skip_filtering=true` 或缺少 `raw/` chunk，會跳過並沿用原始候選。
4. **SAM2 追蹤／補洞**
   - 針對未覆蓋區域進行 mask propagation，並依需求改用 box prompt（`prompt_mode`: `all_mask` / `lt_bbox` / `all_bbox`）。
   - 重要參數：`max_propagate`、`prompt_mode`、`iou_threshold`、`downscale_ratio`、`comparison_sampling`（控制預覽幀 stride / max）。
5. **報告與輸出**
   - `stages.report.enabled=true` 時產出 `report.md`、縮圖、`stage_timings.json`，並搬運 `viz/compare` 預覽。
   - 每輪會寫入 `workflow_summary.json`、`environment_snapshot.json` 與 per-stage 資源峰值。
   - `logs/workflow_history.csv`、`logs/run_pid_map.csv`、`oom_monitor.log` 紀錄執行與資源資訊。
6. **後續檢查**
   - 驗證輸出資料夾（如 `outputs/experiments/<name>/<scene>/<timestamp>_...`），確認 `manifest.json`、`raw/`、`filtered/`、`tracking/` 完整。
   - 檢查 `workflow_summary.json` 中的 `stages.*.resources` 是否合理；必要時比對 `environment_snapshot.json`。
   - 使用 `python -m my3dis.generate_report` 重新產出報告或補漏。

## 6. 常用執行方式
- **推薦：run_experiment.sh（雙階段自動切換環境）**
  ```bash
  cd /path/to/My3DIS
  ./run_experiment.sh --levels 2,4,6 --frames 1200:1600:20 --ssam-freq 2 --sam2-max-propagate 30
  ```
  - `--dry-run` 僅印出命令。
  - `--no-timestamp` 直接覆寫固定輸出資料夾。
  - `--experiment-tag` 追加自訂標籤便於識別。
- **手動雙階段**
  1. Semantic-SAM：
     ```bash
     export PYTHONPATH="$(pwd)/src"
     conda run -n Semantic-SAM \
       python -m my3dis.generate_candidates \
         --data-path ./data/multiscan/scene_00065_00/outputs/color \
         --levels 2,4,6 \
         --frames 1200:1600:20 \
         --ssam-freq 2 \
         --output ./outputs/scene_00065_00
     ```
  2. SAM2：
     ```bash
     TS=$(ls -1dt ./outputs/scene_00065_00/* | head -n1)
     export PYTHONPATH="$(pwd)/src"
     conda run -n SAM2 \
       python -m my3dis.track_from_candidates \
         --run-dir "$TS" \
         --max-propagate 30 \
         --prompt-mode all_mask
     ```
- **YAML 工作流程（批次或多場景）**
  ```bash
  export PYTHONPATH="$(pwd)/src"
  python -m my3dis.run_workflow \
    --config configs/multiscan/base.yaml \
    --override-output ./outputs/experiments/parallel_1009
  ```
  - 可搭配 `scripts/pipeline/run_workflow_batch.py` 進行跨場景批次。

## 7. 輸出結構與驗證
- **資料夾命名**：`YYYYMMDD_HHMMSS_L2_4_6_ssam2_propmax30_<tag>`；可從名稱快速理解參數。
- **關鍵檔案**：
  - `manifest.json`：紀錄 levels、frames、SSAM 頻率、mask 縮放等摘要。
  - `workflow_summary.json` / `environment_snapshot.json`：Stage 耗時、資源峰值與執行環境。
  - `level_{L}/raw/manifest.json` + `chunk_*.tar.gz`：每層原始 SSAM 候選（meta + mask stack）。
  - `level_{L}/filtered/filtered.json`：重新過濾後的遮罩資料（含壓縮遮罩）。
  - `level_{L}/tracking/video_segments*.npz`、`object_segments*.npz`：SAM2 追蹤結果（可能含 `_scale{ratio}x` 後綴）。
  - `level_{L}/tracking/viz/compare/`：比較圖、`*_comparison_summary.json`、無圖時的 fallback JSON。
  - `selected_frames/`：本次處理的影格快照（symlink 或副本）。
  - `report.md` + `images/`：Markdown 報告與縮圖。
  - `stage_timings.json`：各階段耗時（若 `record_timings` 啟用）。
- **執行紀錄**：
  - `logs/workflow_history.csv`：每次 run 概要。
  - `logs/new/run_exp_*.log`：最新實驗詳細 log，含錯誤堆疊。
  - `logs/run_pid_map.csv`：PID 對應與 stdout 路徑，方便追查背景程序。
  - `logs/oom_monitor.log`：cgroup `memory.events` 監控輸出。
  - `logs/workflow_notifications.log`：完成或錯誤摘要訊息。

## 8. 除錯與最佳實務
- **快速定位問題**
  - 檢查 log 尾段是否有 `WorkflowConfigError`（組態問題）或 `WorkflowRuntimeError`（執行例外）。
  - 若 GPU 資源不足，調整 `stages.gpu` 或改用 `ssam_freq` 降低負載。
- **常見排查清單**
  - 掃描 `outputs/.../stage_timings.json` 找出耗時瓶頸。
  - 檢視 `workflow_summary.json` → `stages.*.resources` 了解 CPU/GPU 峰值或缺失監控。
  - 驗證 `selected_frames/` 是否包含預期影格，確認 slice 是否設定正確。
  - 若報告圖片缺失，重新執行 `python -m my3dis.generate_report`。
  - 透過 `MY3DIS_*` 環境變數外掛自訂路徑，不要直接修改程式常數。
- **協作建議**
  - 調整 YAML 前先複製為自訂檔案，避免覆寫共用設定。
  - 大量場景執行時，務必先做小範圍 smoke test。
  - 撰寫新腳本時，沿用 `src/my3dis/common_utils.py` 的 logging 格式保持一致。

## 9. 建議入門順序
1. 閱讀 `README.md` 與本指南，掌握高層流程。
2. 跑一次 `./run_experiment.sh --dry-run` 確認理解指令。
3. 以小場景實際執行，並檢視輸出結構與報告。
4. 研究 `configs/multiscan/base.yaml`，必要時參考實驗性的 `src/my3dis/config.py` 以掌握未來統一配置方向。⭐
5. 深入閱讀 `src/my3dis/run_workflow.py` 與 `workflow/` 模組，了解 Pipeline 擴充點。

## 10. 有問題時怎麼求助
- 先附上：使用的 YAML、指令、log 段落、輸出路徑。
- 將錯誤訊息對照 `logs/new/run_exp_*.log` 與 `logs/workflow_history.csv`。
- 若確認為環境或資料路徑錯誤，請同步更新 `env/README` 或相關變數，同組夥伴才能復現。
- 需要調整流程時，與維護者討論是否應更新 `run_experiment.sh` 或 `configs/index`，避免每人手動 patch。

祝順利上手，有任何改善建議歡迎補充於本檔案或提交 PR！
