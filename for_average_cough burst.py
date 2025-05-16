import os
import numpy as np
import librosa
import matplotlib.pyplot as plt


def count_cough_bursts(file_path, top_db=30):
    """
    Load `file_path`, split on energy valleys,
    and return the number of detected bursts.
    """
    y, sr = librosa.load(file_path, sr=None)
    # split into non-silent intervals
    intervals = librosa.effects.split(y, top_db=top_db)
    return len(intervals)


def analyze_dataset(directory):
    """
    For every audio file in `directory`, count cough bursts.
    Print summary stats and plot a histogram.
    """
    counts = []
    for fname in os.listdir(directory):
        if not fname.lower().endswith(('.wav', '.mp3', '.3gp')):
            continue
        path = os.path.join(directory, fname)
        n = count_cough_bursts(path)
        counts.append(n)
    counts = np.array(counts)

    print("Files analyzed:", len(counts))
    print(f"Min bursts: {counts.min()}")
    print(f"Max bursts: {counts.max()}")
    print(f"Mean bursts: {counts.mean():.2f}")
    print(f"Std dev: {counts.std():.2f}")

    # Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(counts, bins=range(0, counts.max() + 2), edgecolor='black', align='left')
    plt.title("Distribution of Cough Bursts per File")
    plt.xlabel("Number of bursts detected")
    plt.ylabel("Number of files")
    plt.xticks(range(0, counts.max() + 1))
    plt.tight_layout()
    plt.show()


# --- Usage ---
if __name__ == "__main__":
    # change this to your folder
    AUDIO_DIR = r"C:\Users\User\Downloads\organized_cough_dataset\pneumonia"
    analyze_dataset(AUDIO_DIR)
