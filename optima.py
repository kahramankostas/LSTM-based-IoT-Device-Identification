import os
for n in range(17, 21):  # 2 dahil, 16 dahil
    cmd = f"python 02-lstm_pipeline.py --train_csv Train_IoTDevIDv1.csv --val_csv Validation_IoTDevIDv1.csv --test_csv Validation_IoTDevIDv1.csv  --epochs 100 --trials 10 --out_dir ./{n} --seq_len {n}"
    print(f"\n========== seq_len = {n} ==========")
    os.system(cmd)