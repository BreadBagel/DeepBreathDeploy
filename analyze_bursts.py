# analyze_bursts.py

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

# ————— CONFIGURATION —————
AUDIO_DIR = r'C:\Users\User\Downloads\organized_cough_dataset\pneumonia'  # update this path
MIN_DURATION_THRESHOLD = 0.4  # seconds (for "≥ 0.4 s" check)
OUTPUT_CSV = 'burst_stats.csv'
# ——————————————————————

def analyze_cough_bursts(audio_dir, min_duration_thresh):
    """
    Scan each audio file to:
      • detect cough bursts
      • compute burst durations
      • record count, avg duration, and proportion ≥ threshold
    Returns a DataFrame of per-file stats.
    """
    stats = []
    for fname in sorted(os.listdir(audio_dir)):
        if not fname.lower().endswith(('.wav', '.mp3', '.3gp')):
            continue
        path = os.path.join(audio_dir, fname)
        y, sr = librosa.load(path, sr=None)
        # detect non-silent intervals (cough bursts)
        intervals = librosa.effects.split(y, top_db=30)
        durations = [(end - start) / sr for start, end in intervals]

        num_bursts = len(durations)
        mean_dur = float(np.mean(durations)) if durations else 0.0
        prop_ge = float(np.mean([d >= min_duration_thresh for d in durations])) if durations else 0.0

        stats.append({
            'filename': fname,
            'num_bursts': num_bursts,
            'mean_duration_s': mean_dur,
            f'prop_≥{min_duration_thresh}s': prop_ge
        })

    df = pd.DataFrame(stats)
    return df

def main():
    # 1) Analyze
    df = analyze_cough_bursts(AUDIO_DIR, MIN_DURATION_THRESHOLD)

    # 2) Save CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved per-file burst stats to {OUTPUT_CSV}\n")

    # 3) Print overall summary
    total = len(df)
    avg_bursts = df['num_bursts'].mean()
    std_bursts = df['num_bursts'].std()
    avg_dur = df['mean_duration_s'].mean()
    prop_overall = df[f'prop_≥{MIN_DURATION_THRESHOLD}s'].mean()

    print("=== Overall Dataset Summary ===")
    print(f"Total files:           {total}")
    print(f"Avg bursts per file:   {avg_bursts:.2f} ± {std_bursts:.2f}")
    print(f"Avg burst duration:    {avg_dur:.2f} s")
    print(f"Overall % ≥ {MIN_DURATION_THRESHOLD}s:  {prop_overall*100:.1f}%\n")

    # 4) Histogram of burst counts
    plt.figure()
    plt.hist(df['num_bursts'], bins=range(0, int(df['num_bursts'].max())+2), edgecolor='black')
    plt.title("Cough Bursts per File")
    plt.xlabel("Number of bursts detected")
    plt.ylabel("Number of files")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
