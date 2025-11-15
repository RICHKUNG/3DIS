#!/usr/bin/env python3
"""
批量更新配置文件中的层级定义：L2/L4/L6 → L1/L3/L5

使用场景：当实验使用 level_1/3/5 目录结构时，需要更新所有配置文件的层级引用
"""

import re
from pathlib import Path

def fix_config_file(config_path: Path, dry_run: bool = False):
    """修复单个配置文件的层级定义"""
    with open(config_path, 'r') as f:
        content = f.read()

    original_content = content

    # 替换规则
    replacements = [
        # object_levels 和 part_levels
        (r'object_levels:\s*\[L2,\s*L4\]', 'object_levels: [L1, L3]'),
        (r'part_levels:\s*\[L4,\s*L6\]', 'part_levels: [L3, L5]'),

        # available_levels
        (r'available_levels:\s*\[L2,\s*L4,\s*L6\]', 'available_levels: [L1, L3, L5]'),

        # 注释中的 L2 参考
        (r'# Top-K for coarse localization at L2', '# Top-K for coarse localization at L1'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    if content != original_content:
        if not dry_run:
            with open(config_path, 'w') as f:
                f.write(content)
            print(f"✓ Fixed: {config_path}")
        else:
            print(f"Would fix: {config_path}")
        return True
    else:
        print(f"  Skipped (no changes): {config_path}")
        return False

def main():
    import argparse

    parser = argparse.ArgumentParser(description='批量更新配置文件层级定义')
    parser.add_argument('--dry-run', action='store_true', help='预览模式（不实际修改文件）')
    parser.add_argument('--config-dir', default='configs/inference', help='配置文件目录')
    args = parser.parse_args()

    config_dir = Path(__file__).parent.parent / args.config_dir

    if not config_dir.exists():
        print(f"错误：配置目录不存在: {config_dir}")
        return

    print(f"扫描配置目录: {config_dir}")
    print(f"模式: {'预览' if args.dry_run else '修改'}")
    print("=" * 60)

    # 查找所有 YAML 文件
    yaml_files = list(config_dir.glob('*.yaml'))
    fixed_count = 0

    for yaml_file in sorted(yaml_files):
        if fix_config_file(yaml_file, dry_run=args.dry_run):
            fixed_count += 1

    print("=" * 60)
    print(f"完成！修改了 {fixed_count}/{len(yaml_files)} 个文件")

    if args.dry_run:
        print("\n提示：使用不带 --dry-run 参数运行以实际修改文件")

if __name__ == '__main__':
    main()
