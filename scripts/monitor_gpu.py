#!/usr/bin/env python3
"""
GPU 記憶體使用監控腳本
持續監控並記錄 GPU 記憶體使用情況
"""

import time
import subprocess
import argparse
from datetime import datetime


def get_gpu_memory():
    """獲取 GPU 記憶體使用情況"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split('\n')
    except Exception as e:
        return [f"Error: {e}"]


def main():
    parser = argparse.ArgumentParser(description='Monitor GPU memory usage')
    parser.add_argument('--interval', type=int, default=2,
                       help='Monitoring interval in seconds (default: 2)')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Optional log file to save monitoring data')
    parser.add_argument('--alert-threshold', type=float, default=90.0,
                       help='Alert when memory usage exceeds this percentage (default: 90)')

    args = parser.parse_args()

    print(f"🔍 GPU Memory Monitor")
    print(f"📊 Interval: {args.interval}s")
    print(f"🚨 Alert threshold: {args.alert_threshold}%")
    if args.log_file:
        print(f"📝 Logging to: {args.log_file}")
    print("=" * 80)
    print(f"{'Time':<20} {'GPU':<5} {'Used (MB)':<12} {'Total (MB)':<12} {'Usage %':<10} {'GPU %':<8}")
    print("=" * 80)

    log_handle = open(args.log_file, 'a') if args.log_file else None

    try:
        while True:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            gpu_data = get_gpu_memory()

            for line in gpu_data:
                if 'Error' in line:
                    print(f"{timestamp:<20} {line}")
                    continue

                parts = line.split(',')
                if len(parts) >= 4:
                    gpu_idx, mem_used, mem_total, gpu_util = [p.strip() for p in parts]

                    try:
                        mem_used_val = float(mem_used)
                        mem_total_val = float(mem_total)
                        usage_pct = (mem_used_val / mem_total_val) * 100

                        alert = "🚨 ALERT!" if usage_pct >= args.alert_threshold else ""

                        output = (f"{timestamp:<20} GPU{gpu_idx:<2} "
                                f"{mem_used_val:<12.1f} {mem_total_val:<12.1f} "
                                f"{usage_pct:<10.1f} {gpu_util:<8} {alert}")

                        print(output)

                        if log_handle:
                            log_handle.write(f"{output}\n")
                            log_handle.flush()

                    except ValueError:
                        continue

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped by user")
    finally:
        if log_handle:
            log_handle.close()


if __name__ == '__main__':
    main()
