"""
Semantic-SAM 影像分割處理管線（精簡版 CLI）
用於生成指定層級的語義分割遮罩，並統一輸出結構

Author: Rich Kung
Updated: 2025-09-09
"""

# =====================
# 套件匯入
# =====================
import os
import sys
import datetime
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import argparse  # optional, kept for future use
import csv
import subprocess

DEFAULT_SEMANTIC_SAM_ROOT = os.environ.get(
    "SEMANTIC_SAM_ROOT",
    "/media/Pluto/richkung/Semantic-SAM",
)

if DEFAULT_SEMANTIC_SAM_ROOT and DEFAULT_SEMANTIC_SAM_ROOT not in sys.path:
    sys.path.append(DEFAULT_SEMANTIC_SAM_ROOT)

from semantic_sam import (
    prepare_image,
    plot_results,
    build_semantic_sam,
    SemanticSamAutomaticMaskGenerator,
)

# 匯入 auto_generation_inference.py 的後處理函式
from auto_generation_inference import (
    instance_map_to_anns,
)

# =====================
# 參數設定
# =====================
MODEL_TYPE = "L"  # 使用 SwinL 模型
CKPT_PATH = os.path.join(
    DEFAULT_SEMANTIC_SAM_ROOT,
    "checkpoints",
    "swinl_only_sam_many2many.pth",
)
# 預設輸出到專案內統一資料夾（可由 CLI 覆蓋）
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "exp_outputs", "progressive_refinement")

# 新增：檔案命名規範設定
EXPERIMENT_NAME = "experiment"
USE_TIMESTAMP = True
VERBOSE = False  # 控制詳細輸出，預設關閉以降低終端輸出
SAVE_VIZ = True  # 預設不輸出大量 parent/child 視覺化


def console(message, *, important=False):
    """集中控制非關鍵輸出的顯示行為"""
    if important or VERBOSE:
        print(message)

# =====================
# 實驗設定（集中於此編輯即可）
# =====================
SCENE_NAME = "scene_00065_00"
DATA_ROOT = "/media/public_dataset2/multiscan/"
LEVELS = [2, 4, 6]          # 必須遞增，且至少兩個層級
RANGE_STR = "1400:1500:20"  # start:end:step（end 為排他）
MIN_AREA = 200
MAX_MASKS = 2000
NO_TIMESTAMP = False        # True 時不加時間戳記

# 額外匯入（用於避免反覆存檔讀檔）
try:
    import torch
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
except Exception:
    torch = None
    transforms = None
    InterpolationMode = None

# =====================
# 工具函數
# =====================
def timer_decorator(func):
    """計時裝飾器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        
        func_name = func.__name__.replace('_', ' ').title()
        if duration < 60:
            print(f"⏱️  {func_name} 完成，耗時: {duration:.2f} 秒")
        else:
            minutes = int(duration // 60)
            seconds = duration % 60
            print(f"⏱️  {func_name} 完成，耗時: {minutes}分{seconds:.1f}秒")
        
        return result
    return wrapper

def log_step(step_name, start_time=None):
    """記錄步驟進度"""
    if start_time:
        duration = time.time() - start_time
        if duration < 60:
            print(f"✅ {step_name} - 耗時: {duration:.2f}秒")
        else:
            minutes = int(duration // 60)
            seconds = duration % 60
            print(f"✅ {step_name} - 耗時: {minutes}分{seconds:.1f}秒")
    else:
        print(f"🔄 開始: {step_name}")


def parse_levels(levels_str):
    """將 '2,4,6' 解析為 [2,4,6]"""
    if isinstance(levels_str, (list, tuple)):
        return [int(x) for x in levels_str]
    return [int(x) for x in str(levels_str).split(',') if str(x).strip()]


def parse_range(range_str):
    """將 'start:end:step' 解析為三元組"""
    parts = str(range_str).split(':')
    if len(parts) != 3:
        raise ValueError("range 需為 'start:end:step' 格式，例如 1400:1700:20")
    return int(parts[0]), int(parts[1]), int(parts[2])


def get_git_commit_hash(default=None):
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__))
        return out.decode("utf-8").strip()
    except Exception:
        return default


def instance_map_to_color_image(instance_map):
    """將 instance_map 轉為彩色圖（利於快速總覽）"""
    h, w = instance_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    ids = np.unique(instance_map)
    rng = np.random.default_rng(0)
    color_map = {0: (0, 0, 0)}
    for uid in ids:
        if uid == 0:
            continue
        color_map[int(uid)] = tuple(rng.integers(0, 255, size=3).tolist())
    for uid, color in color_map.items():
        colored[instance_map == uid] = color
    return Image.fromarray(colored, 'RGB')


def bbox_from_mask(seg):
    ys, xs = np.where(seg)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

# =====================
# 核心功能函數
# =====================
def get_experiment_timestamp():
    """生成實驗時間戳記"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def create_experiment_folder(base_path, experiment_name, timestamp=None):
    """建立帶時間戳記的實驗資料夾"""
    if timestamp is None:
        timestamp = get_experiment_timestamp()
    
    experiment_folder = f"{experiment_name}_{timestamp}"
    full_path = os.path.join(base_path, experiment_folder)
    os.makedirs(full_path, exist_ok=True)
    
    return full_path, timestamp

def setup_output_directories(experiment_path):
    """建立輸出目錄結構（精簡統一）
    experiment_path/
      original/
      levels/level_{L}/
      relations/
      summary/
      viz/ (可選)
    """
    dirs = [
        "original",
        "levels",
        "relations",
        "summary",
        "viz",
    ]
    created_dirs = {}
    for d in dirs:
        p = os.path.join(experiment_path, d)
        os.makedirs(p, exist_ok=True)
        created_dirs[d] = p
    return created_dirs

@timer_decorator
def save_original_image_info(image_path, output_dirs):
    """儲存原始圖片"""
    try:
        original_dir = output_dirs["original"]
        
        # 複製原始圖片到輸出目錄
        import shutil
        original_filename = os.path.basename(image_path)
        copied_path = os.path.join(original_dir, f"original_{original_filename}")
        shutil.copy2(image_path, copied_path)
        
        return {"original_path": image_path, "copied_path": copied_path}
    except Exception as e:
        console(f"❌ 儲存原始圖片時發生錯誤: {str(e)}", important=True)
        return None


def prepare_image_from_pil(pil_img):
    """仿照 semantic_sam.prepare_image，但接受 PIL 物件，避免落地存檔再讀取。
    回傳 (image_ori_np, torch_tensor_on_cuda)
    """
    if transforms is None or torch is None:
        # 後備：落地存檔（較慢，盡量避免）
        tmp_path = os.path.join("/tmp", f"masked_{int(time.time()*1000)}.png")
        try:
            pil_img.save(tmp_path)
            return prepare_image(image_pth=tmp_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    resize_interpolation = InterpolationMode.BICUBIC if InterpolationMode else Image.BICUBIC
    transform1 = transforms.Resize(640, interpolation=resize_interpolation)
    image_resized = transform1(pil_img)
    image_ori = np.asarray(image_resized)
    images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()
    return image_ori, images

def create_masked_image(original_image, mask_data, background_color=(0, 0, 0)):
    """
    創建被 mask 的圖片，mask 外的區域設為指定背景色
    
    Args:
        original_image: PIL Image 原始圖片
        mask_data: numpy array mask 數據 (True/False 或 0/255)
        background_color: 背景填充顏色 (R, G, B)
        
    Returns:
        PIL Image: 被 mask 的圖片
    """
    try:
        # 確保輸入圖片是 PIL Image 且為 RGB 模式
        if not isinstance(original_image, Image.Image):
            raise ValueError("輸入必須是 PIL Image")
        
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        # 將圖片轉為 numpy array
        img_array = np.array(original_image)
        result_array = img_array.copy()
        
        # 處理 mask 數據
        if isinstance(mask_data, np.ndarray):
            # 標準化 mask 為 boolean
            if mask_data.dtype == bool:
                mask_binary = mask_data
            elif mask_data.dtype == np.uint8:
                mask_binary = mask_data > 0
            else:
                mask_binary = mask_data > 0.5
            
            # 確保 mask 是 2D
            if len(mask_binary.shape) > 2:
                mask_binary = mask_binary.squeeze()
                if len(mask_binary.shape) > 2:
                    mask_binary = mask_binary[:, :, 0]
            
            # 調整 mask 尺寸以匹配圖片
            if mask_binary.shape != img_array.shape[:2]:
                mask_pil = Image.fromarray((mask_binary * 255).astype(np.uint8), 'L')
                mask_pil = mask_pil.resize(original_image.size, Image.NEAREST)
                mask_binary = np.array(mask_pil) > 127
            
            # 將 mask 外的區域設為背景色
            mask_inverse = ~mask_binary
            for c in range(3):  # RGB 通道
                result_array[mask_inverse, c] = background_color[c]
            
            return Image.fromarray(result_array, 'RGB')
        else:
            if VERBOSE:
                console(f"警告：mask 數據格式不正確: {type(mask_data)}")
            return original_image
            
    except Exception as e:
        if VERBOSE:
            console(f"創建 masked 圖片時發生錯誤: {str(e)}")
        return original_image

@timer_decorator
def progressive_refinement_masks(semantic_sam, image_path, level_sequence, output_dirs, min_area=50, max_masks_per_level=200, save_viz=False):
    """
    執行漸進式細化 mask 生成，並以有序方式儲存結果
    """
    main_start = time.time()
    log_step(f"漸進式細化處理 (Levels: {level_sequence})")
    
    # 驗證 level 序列
    if not level_sequence or len(level_sequence) < 2:
        raise ValueError("level_sequence 必須包含至少 2 個 level")
    
    if level_sequence != sorted(level_sequence):
        raise ValueError("level_sequence 必須是遞增序列")
    
    # 載入原始圖片
    original_image_pil, input_image = prepare_image(image_pth=image_path)
    
    # 確保 original_image_pil 是 PIL Image 格式
    if not isinstance(original_image_pil, Image.Image):
        if isinstance(original_image_pil, np.ndarray):
            # 轉換 numpy array 為 PIL Image
            if len(original_image_pil.shape) == 3:
                if original_image_pil.shape[2] == 3:  # RGB
                    if original_image_pil.dtype != np.uint8:
                        original_image_pil = (original_image_pil * 255).astype(np.uint8)
                    original_image_pil = Image.fromarray(original_image_pil, 'RGB')
                elif original_image_pil.shape[2] == 4:  # RGBA
                    if original_image_pil.dtype != np.uint8:
                        original_image_pil = (original_image_pil * 255).astype(np.uint8)
                    original_image_pil = Image.fromarray(original_image_pil, 'RGBA').convert('RGB')
            elif len(original_image_pil.shape) == 2:  # 灰階
                if original_image_pil.dtype != np.uint8:
                    original_image_pil = (original_image_pil * 255).astype(np.uint8)
                original_image_pil = Image.fromarray(original_image_pil, 'L').convert('RGB')
            console(f"已將 numpy array 轉換為 PIL Image: {original_image_pil.size}")
        else:
            raise TypeError(f"不支援的圖片格式: {type(original_image_pil)}")
    
    # 確保是 RGB 模式
    if original_image_pil.mode != 'RGB':
        original_image_pil = original_image_pil.convert('RGB')
    
    console(f"原始圖片格式確認: {type(original_image_pil)}, 模式: {original_image_pil.mode}, 尺寸: {original_image_pil.size}")
    
    # 設定目錄
    sequence_str = "_".join(map(str, level_sequence))
    parent_child_dir = os.path.join(output_dirs["viz"], f"sequence_{sequence_str}")
    levels_root = os.path.join(output_dirs["levels"], f"sequence_{sequence_str}")
    os.makedirs(levels_root, exist_ok=True)
    if save_viz:
        os.makedirs(parent_child_dir, exist_ok=True)
    
    # 只保留必要的結果變數
    refinement_results = {
        'level_sequence': level_sequence,
        'levels': {}
    }
    tree_relations = {}  # 每層的節點列表
    id_to_node = {}      # 全域唯一 id 對應 node
    mask_id_counter = 1  # 全域唯一 id
    all_nodes = []       # 新增：收集所有節點

    # 第一個 level：在原圖上生成 mask
    first_level = level_sequence[0]
    console(f"\n🎯 Level {first_level} (原圖)")

    mask_generator = SemanticSamAutomaticMaskGenerator(semantic_sam, level=[first_level])
    first_level_masks = mask_generator.generate(input_image)

    # === 合併為 instance map，拆回 mask list，確保同層不重疊 ===
    height, width = original_image_pil.size[1], original_image_pil.size[0]
    instance_map = np.zeros((height, width), dtype=np.int32)
    for idx, mask in enumerate(first_level_masks):
        if 'segmentation' in mask:
            seg = mask['segmentation']
            instance_map[(seg) & (instance_map == 0)] = idx + 1  # id從1開始
    first_level_masks = instance_map_to_anns(instance_map)

    # 先不儲存，待 unique_id 指派後用 unique_id 重新建圖再存

    # === 唯一 id 與樹狀結構 ===
    tree_relations[first_level] = []
    # 第一層
    for mask in first_level_masks:
        node = {
            "id": mask_id_counter,
            "parent": None,
            "children": []
        }
        tree_relations[first_level].append(node)
        id_to_node[mask_id_counter] = node
        all_nodes.append(node)  # 收集
        mask['unique_id'] = mask_id_counter
        mask_id_counter += 1

    # 過濾和限制 mask 數量
    first_level_masks = [m for m in first_level_masks if m.get('area', 0) >= min_area]
    if len(first_level_masks) > max_masks_per_level:
        first_level_masks = first_level_masks[:max_masks_per_level]

    console(f"✨ Level {first_level}: {len(first_level_masks)} 個有效 mask")
    
    # 使用 unique_id 建立 instance_map 並儲存
    level_dir = os.path.join(levels_root, f"level_{first_level}")
    os.makedirs(level_dir, exist_ok=True)
    instance_map_unique = np.zeros((height, width), dtype=np.int32)
    for m in first_level_masks:
        if 'segmentation' in m:
            seg = m['segmentation']
            instance_map_unique[(seg) & (instance_map_unique == 0)] = m['unique_id']
    np.save(os.path.join(level_dir, "instance_map.npy"), instance_map_unique)
    try:
        instance_img = instance_map_to_color_image(instance_map_unique)
        instance_img.save(os.path.join(level_dir, "colored_instances.png"))
    except Exception:
        pass

    # 儲存第一個 level 的視覺化結果（原生函式）
    plot_results(first_level_masks, original_image_pil, save_path=level_dir)
        
    refinement_results['levels'][first_level] = {
        'masks': first_level_masks,
        'mask_count': len(first_level_masks),
    }

    # 當前處理的 mask 列表（用於下一個 level 的輸入）
    current_masks = first_level_masks
    current_level = first_level
    
    # 處理後續的 level
    level_pbar = tqdm(level_sequence[1:], desc="漸進細化", bar_format='{l_bar}{bar:20}{r_bar}')
    
    for next_level in level_pbar:
        level_start = time.time()
        level_pbar.set_description(f"Level {current_level}→{next_level}")
        
        next_level_all_masks = []
        next_level_parent_info = []
        
        # 為本層建立 parent-child 關係目錄
        # 更直觀的階層資料夾命名：L{cur}_to_L{next}
        parent_child_level_dir = os.path.join(parent_child_dir, f"L{current_level}_to_L{next_level}")
        if save_viz:
            os.makedirs(parent_child_level_dir, exist_ok=True)
        
        # 為每個當前 level 的 mask 生成更細的 mask
        parent_pbar = tqdm(enumerate(current_masks), total=len(current_masks), 
                          desc="處理 parent masks", leave=False,
                          bar_format='{l_bar}{bar:15}{r_bar}{postfix}')
        
        successful_parents = 0
        total_children = 0
        
        for parent_idx, parent_mask in parent_pbar:
            try:
                # 獲取 parent mask 的 segmentation 數據
                if 'segmentation' not in parent_mask:
                    next_level_parent_info.append({
                        'parent_id': parent_idx,
                        'child_count': 0,
                        'error': 'no_segmentation'
                    })
                    continue
                
                parent_seg = parent_mask['segmentation']
                
                # 創建被 mask 的圖片（mask 外為黑色）
                masked_image = create_masked_image(original_image_pil, parent_seg, background_color=(0, 0, 0))
                
                # 確保 masked_image 是 PIL Image
                if not isinstance(masked_image, Image.Image):
                    next_level_parent_info.append({
                        'parent_id': parent_idx,
                        'child_count': 0,
                        'error': 'masked_image_format_error'
                    })
                    continue
                
                # 可選儲存 masked 圖片
                if save_viz:
                    try:
                        parent_uid = current_masks[parent_idx]['unique_id']
                        parent_dir = os.path.join(parent_child_level_dir, f"P{parent_uid}")
                        os.makedirs(parent_dir, exist_ok=True)
                        masked_img_path = os.path.join(parent_dir, f"parent_L{current_level}_P{parent_uid}.png")
                        masked_image.save(masked_img_path)
                    except Exception as save_error:
                        if VERBOSE:
                            console(f"儲存 masked 圖片失敗: {str(save_error)}")

                # 直接將 masked PIL 轉換為模型輸入格式（避免先落地）
                try:
                    _, masked_input = prepare_image_from_pil(masked_image)
                except Exception as _:
                    next_level_parent_info.append({
                        'parent_id': parent_idx,
                        'child_count': 0,
                        'error': 'prepare_image_failed'
                    })
                    continue
                
                # 在 masked 圖片上生成下一個 level 的 mask
                try:
                    mask_generator = SemanticSamAutomaticMaskGenerator(semantic_sam, level=[next_level])
                    child_masks = mask_generator.generate(masked_input)
                    # 🔥 child segmentation 與 parent segmentation 做 AND
                    for child_mask in child_masks:
                        if 'segmentation' in child_mask:
                            child_mask['segmentation'] = np.logical_and(child_mask['segmentation'], parent_seg)
                    # 🔥 只保留面積大於 min_area 且 segmentation 不等於 parent_seg 的 child mask
                    def mask_iou(mask1, mask2):
                        inter = np.logical_and(mask1, mask2).sum()
                        union = np.logical_or(mask1, mask2).sum()
                        return inter / union if union > 0 else 0.0
                    similarity_threshold = 0.95
                    child_masks = [
                        m for m in child_masks
                        if np.sum(m['segmentation']) >= min_area and mask_iou(m['segmentation'], parent_seg) < similarity_threshold
                    ]
                except Exception as _:
                    next_level_parent_info.append({
                        'parent_id': parent_idx,
                        'child_count': 0,
                        'error': 'mask_generation_failed'
                    })
                    continue
                
                # 過濾有效的 child mask
                valid_child_masks = []
                if save_viz:
                    parent_uid = current_masks[parent_idx]['unique_id']
                    child_dir = os.path.join(parent_child_level_dir, f"P{parent_uid}")
                    os.makedirs(child_dir, exist_ok=True)

                for child_idx, child_mask in enumerate(child_masks):
                    if child_mask.get('area', 0) >= min_area:
                        # 添加 parent 資訊
                        child_mask['parent_mask_id'] = parent_idx
                        child_mask['parent_level'] = current_level
                        child_mask['current_level'] = next_level
                        # 分配唯一 id
                        child_mask['unique_id'] = mask_id_counter
                        valid_child_masks.append(child_mask)
                        # 建立 parent-child 關聯
                        parent_unique_id = current_masks[parent_idx]['unique_id']
                        mask_id_counter += 1
                        # 如果有分割資訊，儲存 child mask 的視覺化
                        if save_viz and 'segmentation' in child_mask:
                            try:
                                child_mask_img = create_masked_image(original_image_pil, child_mask['segmentation'], background_color=(0, 0, 0))
                                parent_uid = current_masks[parent_idx]['unique_id']
                                child_uid = child_mask.get('unique_id', mask_id_counter)
                                child_img_path = os.path.join(child_dir, f"child_L{next_level}_C{child_uid}_from_P{parent_uid}.png")
                                child_mask_img.save(child_img_path)
                            except Exception as save_error:
                                if VERBOSE:
                                    console(f"儲存 child 圖片失敗: {str(save_error)}")

                # 如果成功生成了 child masks，存儲親子關係圖片（移到這裡）
                if save_viz and valid_child_masks:
                    try:
                        parent_uid = current_masks[parent_idx]['unique_id']
                        parent_img_path = os.path.join(child_dir, f"parent_L{current_level}_P{parent_uid}.png")
                        masked_image.save(parent_img_path)
                    except Exception:
                        pass

                next_level_all_masks.extend(valid_child_masks)
                next_level_parent_info.append({
                    'parent_id': parent_idx,
                    'child_count': len(valid_child_masks),
                    'error': None
                })
                
                if len(valid_child_masks) > 0:
                    successful_parents += 1
                    total_children += len(valid_child_masks)
                
                # 記錄parent與child對應
                
                # 更新進度條
                avg_children = total_children / successful_parents if successful_parents > 0 else 0
                parent_pbar.set_postfix({
                    "成功": successful_parents,
                    "平均子數": f"{avg_children:.1f}"
                })
                
            except Exception as e:
                next_level_parent_info.append({
                    'parent_id': parent_idx,
                    'child_count': 0,
                    'error': str(e)
                })
                continue
        
        parent_pbar.close()

        # 合併所有 child mask 為 instance map，拆回 mask list
        if next_level_all_masks:
            height, width = original_image_pil.size[1], original_image_pil.size[0]
            instance_map = np.zeros((height, width), dtype=np.int32)
            for idx, mask in enumerate(next_level_all_masks):
                if 'segmentation' in mask:
                    seg = mask['segmentation']
                    instance_map[(seg) & (instance_map == 0)] = idx + 1
            # instance_map_to_anns 產生的新 mask 需重新分配唯一 id 與 parent
            new_masks = instance_map_to_anns(instance_map)
            new_mask_list = []
            for idx, mask in enumerate(new_masks):
                # 嘗試找出來源 parent
                # 這裡用最重疊的原始 child mask 來決定 parent
                max_iou = 0
                parent_unique_id = None
                for orig_mask in next_level_all_masks:
                    inter = np.logical_and(mask['segmentation'], orig_mask['segmentation']).sum()
                    union = np.logical_or(mask['segmentation'], orig_mask['segmentation']).sum()
                    iou = inter / union if union > 0 else 0
                    if iou > max_iou:
                        max_iou = iou
                        parent_unique_id = orig_mask.get('parent_mask_id', None)
                        parent_unique_id = orig_mask.get('parent_unique_id', orig_mask.get('parent_mask_id', None))
                        # 這裡 parent_unique_id 其實應該是 parent 的 unique_id
                        if 'parent_unique_id' in orig_mask:
                            parent_unique_id = orig_mask['parent_unique_id']
                        else:
                            # 從 current_masks 取得
                            if orig_mask.get('parent_mask_id') is not None and orig_mask['parent_mask_id'] < len(current_masks):
                                parent_unique_id = current_masks[orig_mask['parent_mask_id']]['unique_id']
                # 若找不到，預設指派到上一層第一個
                if parent_unique_id is None and len(current_masks) > 0:
                    parent_unique_id = current_masks[0]['unique_id']
                # 分配唯一 id
                mask['unique_id'] = mask_id_counter
                mask['parent_unique_id'] = parent_unique_id
                # 建立 parent-child 關聯
                mask_id_counter += 1
                new_mask_list.append(mask)
            next_level_all_masks = new_mask_list
            for m in next_level_all_masks:
                m['current_level'] = next_level

        # 使用 unique_id 重新建立 instance_map 並儲存（避免重複且更直觀）
        next_level_dir = os.path.join(levels_root, f"level_{next_level}")
        os.makedirs(next_level_dir, exist_ok=True)
        instance_map_unique = np.zeros((height, width), dtype=np.int32)
        for m in next_level_all_masks:
            if 'segmentation' in m:
                seg = m['segmentation']
                instance_map_unique[(seg) & (instance_map_unique == 0)] = m['unique_id']
        np.save(os.path.join(next_level_dir, "instance_map.npy"), instance_map_unique)
        try:
            instance_img = instance_map_to_color_image(instance_map_unique)
            instance_img.save(os.path.join(next_level_dir, "colored_instances.png"))
        except Exception:
            pass

        # 建立本層樹狀關係
        tree_relations[next_level] = []
        for mask in next_level_all_masks:
            node = {
                "id": mask['unique_id'],
                "parent": mask.get('parent_unique_id', None),
                "children": []
            }
            tree_relations[next_level].append(node)
            id_to_node[node["id"]] = node
            all_nodes.append(node)  # 收集

        # 補充 parent 的 children
        for node in tree_relations[next_level]:
            pid = node["parent"]
            if pid is not None and pid in id_to_node:
                # 避免重複加入
                if node["id"] not in id_to_node[pid]["children"]:
                    id_to_node[pid]["children"].append(node["id"])

        # 單層樹狀關係（JSON）
        with open(os.path.join(next_level_dir, "tree.json"), "w", encoding="utf-8") as f:
            json.dump(tree_relations[next_level], f, ensure_ascii=False, indent=2)

        refinement_results['levels'][next_level] = {
            'masks': next_level_all_masks,
            'mask_count': len(next_level_all_masks),
        }

        current_masks = next_level_all_masks
        current_level = next_level
        # 顯示本層統計
        level_duration = time.time() - level_start
        console(
            f"✨ Level {next_level}: {len(next_level_all_masks)} 個 mask "
            f"(來自 {successful_parents}/{len(next_level_parent_info)} 個 parent，"
            f"耗時 {level_duration:.1f}秒)",
            important=True,
        )
    
    level_pbar.close()

    # 儲存第一層樹狀關係 JSON
    first_level_dir = os.path.join(levels_root, f"level_{first_level}")
    with open(os.path.join(first_level_dir, "tree.json"), "w", encoding="utf-8") as f:
        json.dump(tree_relations[first_level], f, ensure_ascii=False, indent=2)

    # 儲存全域樹狀關係 JSON（統一放 relations/）
    relations_dir = output_dirs["relations"]
    with open(os.path.join(relations_dir, "tree.json"), "w", encoding="utf-8") as f:
        json.dump(tree_relations, f, ensure_ascii=False, indent=2)

    # 產出簡易關係表（relations.csv）：id,parent,level,area,bbox
    csv_path = os.path.join(relations_dir, "relations.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["id", "parent", "level", "area", "bbox"])  # bbox: x_min,y_min,x_max,y_max
        for level, nodes in tree_relations.items():
            # 從 refinement_results 取 area 與 bbox
            masks = refinement_results['levels'].get(level, {}).get('masks', [])
            # 建立 id -> mask 對應
            id2mask = {m.get('unique_id'): m for m in masks}
            for node in nodes:
                m = id2mask.get(node['id'])
                area = int(m.get('area', 0)) if m else 0
                bbox = None
                if m and 'segmentation' in m:
                    b = bbox_from_mask(m['segmentation'])
                    bbox = b if b else None
                writer.writerow([
                    node['id'], node['parent'], level, area,
                    ("%d,%d,%d,%d" % bbox) if bbox else ""
                ])

    log_step("漸進式細化處理完成", main_start)
    return refinement_results


# =====================
# 主要執行流程
# =====================
@timer_decorator
def main():
    """主要執行流程（設定集中於檔案內）"""
    console("🚀 啟動 Semantic-SAM 影像分割處理管線", important=True)
    # 設定工作目錄到 Semantic-SAM 資料夾
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    console(f"📁 工作目錄: {os.getcwd()}")

    # 初始化模型
    console("🔧 正在載入 Semantic-SAM 模型...")
    model_start = time.time()
    try:
        semantic_sam = build_semantic_sam(model_type=MODEL_TYPE, ckpt=CKPT_PATH)
        model_time = time.time() - model_start
        console(f"✅ 模型載入完成！耗時: {model_time:.2f}秒", important=True)
    except Exception as e:
        console(f"❌ 模型載入失敗: {str(e)}", important=True)
        return

    # 建立實驗資料夾
    setup_start = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    experiment_path, ts = create_experiment_folder(
        OUTPUT_ROOT,
        EXPERIMENT_NAME,
        timestamp=get_experiment_timestamp() if (USE_TIMESTAMP and not NO_TIMESTAMP) else None
    )
    console(f"📂 實驗資料夾: {experiment_path}")
    image_index_csv = os.path.join(experiment_path, 'summary', 'index.csv')

    # 設定輸出目錄
    _ = setup_output_directories(experiment_path)
    setup_time = time.time() - setup_start
    console(f"✅ 目錄設置完成，耗時: {setup_time:.2f}秒", important=True)

    # Manifest（提高可重現性）
    manifest = {
        "timestamp": ts,
        "scene": SCENE_NAME,
        "levels": LEVELS,
        "range": RANGE_STR,
        "min_area": MIN_AREA,
        "max_masks": MAX_MASKS,
        "model_type": MODEL_TYPE,
        "ckpt": CKPT_PATH,
        "git_commit": get_git_commit_hash(),
        "save_viz": bool(SAVE_VIZ),
    }
    with open(os.path.join(experiment_path, 'summary', 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 多張圖片處理
    data_folder_source = os.path.join(DATA_ROOT, SCENE_NAME, "outputs", "color")
    all_color_images = sorted(os.listdir(data_folder_source))
    start_idx, end_idx, frequency = parse_range(RANGE_STR)
    selected_images = [all_color_images[i] for i in range(start_idx, end_idx, frequency) if 0 <= i < len(all_color_images)]
    console(f"共選取 {len(selected_images)} 張圖片進行實驗")

    # 準備 index 彙整
    if not os.path.exists(image_index_csv):
        with open(image_index_csv, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.writer(cf)
            writer.writerow(["image", "image_path", "total_masks", "levels", "output_dir"])

    for idx, selected_image in enumerate(selected_images):
        image_path = os.path.join(data_folder_source, selected_image)
        image_stem = os.path.splitext(selected_image)[0]
        console(f"\n=== 處理第 {idx+1} 張: {image_path} ===")
        # 為每張圖片建立獨立的 output 資料夾
        image_output_root = os.path.join(experiment_path, image_stem)
        os.makedirs(image_output_root, exist_ok=True)
        image_output_dirs = setup_output_directories(image_output_root)
        try:
            # 儲存原始圖片
            save_original_image_info(image_path, image_output_dirs)

            # 漸進式細化
            console("\n🎯 執行漸進式細化")
            level_seq = LEVELS
            refinement_results = progressive_refinement_masks(
                semantic_sam,
                image_path,
                level_sequence=level_seq,
                output_dirs=image_output_dirs,
                min_area=MIN_AREA,
                max_masks_per_level=MAX_MASKS,
                save_viz=bool(SAVE_VIZ)
            )

            # 顯示最終統計 + 輸出 metrics.json
            console("\n📊 === 最終結果統計 ===", important=True)
            total_masks = 0
            metrics = {"levels": {}, "total_masks": 0}
            for level, data in refinement_results['levels'].items():
                mask_count = data['mask_count']
                total_masks += mask_count
                metrics["levels"][str(level)] = {"mask_count": mask_count}
                console(f"Level {level}: {mask_count:4d} 個 mask", important=True)
            metrics["total_masks"] = total_masks
            with open(os.path.join(image_output_root, 'summary', 'metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            # 更新 index.csv
            with open(image_index_csv, 'a', newline='', encoding='utf-8') as cf:
                writer = csv.writer(cf)
                writer.writerow([image_stem, image_path, total_masks, ",".join(map(str, level_seq)), image_output_root])

            console(f"\n🎉 總計生成: {total_masks} 個 mask", important=True)
            console(f"\n✅ 實驗完成！所有結果已儲存到: {image_output_root}", important=True)

        except FileNotFoundError:
            console(f"❌ 錯誤：找不到圖片檔案 {image_path}", important=True)
            console("請檢查圖片路徑是否正確", important=True)
        except Exception as e:
            console(f"❌ 處理過程中發生錯誤: {str(e)}", important=True)
            if VERBOSE:
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
