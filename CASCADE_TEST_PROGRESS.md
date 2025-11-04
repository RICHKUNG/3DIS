# Cascade Filtering Test Progress

**Test Configuration**: `configs/multiscan/test_cascade.yaml`
**Start Time**: 2025-11-04 20:19:35
**Scene**: scene_00005_00

## Configuration

```yaml
cascade_filtering: true
stability_threshold: null  # Area-based filtering only
min_area: 100
ssam_freq: 4
frames: 0:-1:50 (196 frames, 49 SSAM frames)
```

## Expected Timeline

1. **SSAM Stage**: ~15 minutes (49 frames × 3 levels)
2. **Filter Stage**: ~1 minute
3. **SAM2 Tracking**: ~20 minutes (196 frames × 3 levels)
4. **Family Tree Building**: ~1 minute
5. **Visualization**: ~2 minutes (10 families × 3 frames)

**Total Expected**: ~40 minutes

## Progress Updates

### 20:19 - Test Started
- Workflow process PID: 2053446
- Scene worker PID: 2053524
- SSAM stage beginning

### 20:20 - SSAM In Progress
- Processing frame 0.png (Level 2)
- Gap-fill: 4 masks added

### 20:21 - SSAM Continuing
- Processing frame 200.png (Level 2)
- Gap-fill: 4 masks added

---

**Status**: 🔄 RUNNING - SSAM Stage
**Current Frame**: ~200/9600 (2%)
**Estimated Completion**: ~21:00

---

## Monitoring Commands

```bash
# Check process status
ps aux | grep "[r]un_workflow.py.*test_cascade"

# View recent log entries
tail -50 logs/scene_workers/test_cascade_20251104_201936/000_scene_00005_00_pid2053524.log

# Wait for completion and see results
bash scripts/wait_and_report_cascade_test.sh
```

## Comparison Baseline

Previous test results (test_1104_ssam4_filter100_fill50_2):
- scene_00005_01: 942 objects, 194 families, 45 orphans (4.8%)
- scene_00006_00: 2283 objects, 292 families, 34 orphans (1.5%)

Expected improvement with cascade filtering:
- **Orphan rate**: 4.8% → <2%
- **Family completeness**: Higher percentage of L2+L4+L6 families
- **Visualization success**: More families with common frames
