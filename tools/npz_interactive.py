import numpy as np
import os
import sys

# 載入 npz 檔案
npz_file = "/media/Pluto/richkung/My3DIS/outputs/scene_00065_00/20250927_122525_L2_4_6_ssam2_area0_fill10000_no_fill_unlimited/level_6/tracking/object_segments_scale0.3x.npz"
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