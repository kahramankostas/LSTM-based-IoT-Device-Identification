import os

cmd = f"python final_evaluation_and_stats.py --model_path ./18/lstm_model.pt --test_csv Test_IoTDevIDv1.csv --seq_len  18 --iterations 30 --out_dir ./final_18_results"
print(f"\n========== seq_len = 18 Final  ==========")
os.system(cmd)