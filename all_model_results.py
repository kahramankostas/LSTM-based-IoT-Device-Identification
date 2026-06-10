import os
for n in range(2, 21):  # 2 dahil, 16 dahil
    cmd = f"python final_evaluation_and_stats.py --model_path ./{n}/lstm_model.pt --test_csv Validation_IoTDevIDv1.csv --seq_len {n} --iterations 30"
    print(f"\n========== seq_len = {n} ==========")
    os.system(cmd)


