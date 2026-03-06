
import os
import shutil
import glob

def reset_status():
    data_dir = "data"
    hourly_dir = os.path.join(data_dir, "hourly")
    backup_dir = "data_backup_" + os.urandom(4).hex()
    
    print(f"Backing up current data to {backup_dir}...")
    
    # Create backup dirs
    os.makedirs(os.path.join(backup_dir, "daily"), exist_ok=True)
    os.makedirs(os.path.join(backup_dir, "hourly"), exist_ok=True)
    
    # 1. Move Daily CSVs
    daily_files = glob.glob(os.path.join(data_dir, "*.csv"))
    moved_daily = 0
    for f in daily_files:
        filename = os.path.basename(f)
        try:
            shutil.move(f, os.path.join(backup_dir, "daily", filename))
            moved_daily += 1
        except Exception as e:
            print(f"Error moving {filename}: {e}")

    # 2. Move Hourly CSVs
    if os.path.exists(hourly_dir):
        hourly_files = glob.glob(os.path.join(hourly_dir, "*.csv"))
        moved_hourly = 0
        for f in hourly_files:
            filename = os.path.basename(f)
            try:
                shutil.move(f, os.path.join(backup_dir, "hourly", filename))
                moved_hourly += 1
            except Exception as e:
                print(f"Error moving {filename}: {e}")
    else:
        moved_hourly = 0

    print(f"\nSummary:")
    print(f"  Moved {moved_daily} daily files")
    print(f"  Moved {moved_hourly} hourly files")
    print(f"  Backup location: {backup_dir}")
    print("\nAll stocks should now appear as 'Undownloaded' in Control Center.")

if __name__ == "__main__":
    reset_status()
