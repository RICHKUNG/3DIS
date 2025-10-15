import numpy as np
import os
import sys

# Load the NPZ file (prefer CLI argument, fall back to environment or sample path).
npz_file = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "MY3DIS_NPZ_PATH",
    "outputs/scene_00065_00/sample_run/level_6/tracking/object_segments_scale0.3x.npz",
)

if not os.path.exists(npz_file):
    raise SystemExit(f"NPZ file not found: {npz_file}")

data = np.load(npz_file, allow_pickle=True)

# 顯示檔案中的所有鍵
print("Available keys in the npz file:")
for key in data.files:
    print(f"  - {key}")

# 如果只有一個鍵，直接使用它
if len(data.files) == 1:
    key = data.files[0]
    print(f"\nUsing the only available key: '{key}'")
    arr = data[key]
else:
    # 如果有多個鍵，讓用戶選擇
    print(f"\nWhich key would you like to explore? ({'/'.join(data.files)})")
    key = input("Enter key name: ").strip()
    if key in data.files:
        arr = data[key]
    else:
        print(f"Key '{key}' not found. Using the first key: '{data.files[0]}'")
        arr = data[data.files[0]]

# 顯示資料資訊
print(f"\nData type: {type(arr)}")
print(f"Data shape: {arr.shape if hasattr(arr, 'shape') else 'N/A'}")
print(f"Data dtype: {arr.dtype if hasattr(arr, 'dtype') else 'N/A'}")

# 如果是單一物件，嘗試解包
if arr.shape == () and arr.dtype == object:
    obj = arr.item()
    print(f"\nUnpacked object type: {type(obj)}")
    if hasattr(obj, 'keys'):
        print("Object keys:", list(obj.keys()))
else:
    obj = arr

obj = arr.item()
print("\nData loaded successfully. Available in variables: 'data', 'arr', 'obj'")
