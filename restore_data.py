
import os
import shutil
import glob

def restore_data():
    backup_dir = "data_backup_dd43df18"  # Hardcoded from previous step
    data_dir = "data"
    hourly_dir = os.path.join(data_dir, "hourly")
    
    print(f"Restoring data from {backup_dir} to {data_dir}...")
    
    # Ensure target dirs exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(hourly_dir, exist_ok=True)
    
    # 1. Restore Daily CSVs (from backup/daily -> data/)
    daily_files = glob.glob(os.path.join(backup_dir, "daily", "*.csv"))
    restored_daily = 0
    for f in daily_files:
        filename = os.path.basename(f)
        try:
            shutil.move(f, os.path.join(data_dir, filename))
            restored_daily += 1
        except Exception as e:
            print(f"Error restoring {filename}: {e}")

    # 2. Restore Hourly CSVs (from backup/hourly -> data/hourly/)
    hourly_files = glob.glob(os.path.join(backup_dir, "hourly", "*.csv"))
    restored_hourly = 0
    for f in hourly_files:
        filename = os.path.basename(f)
        try:
            shutil.move(f, os.path.join(hourly_dir, filename))
            restored_hourly += 1
        except Exception as e:
            print(f"Error restoring {filename}: {e}")

    print(f"\nRestoration Summary:")
    print(f"  Restored {restored_daily} daily files")
    print(f"  Restored {restored_hourly} hourly files")
    
    # Cleanup empty backup dir
    try:
        shutil.rmtree(backup_dir)
        print(f"  Removed empty backup directory: {backup_dir}")
    except Exception as e:
        print(f"  Could not remove backup dir: {e}")

if __name__ == "__main__":
    restore_data()
