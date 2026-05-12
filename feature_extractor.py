# Import necessary libraries
import os
import json
import librosa
import numpy as np
import pandas as pd

# Paths
DATASET_PATH = "genres_original"
CSV_PATH = "features.csv"
JSON_PATH = "data.json"   # optional

# Audio constants
SAMPLE_RATE = 22050
TRACK_DURATION_SECONDS = 30
NUM_SEGMENTS = 10
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION_SECONDS

# Feature constants
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512


def process_dataset(dataset_path, csv_path, save_json=False):

    data = {
        "mapping": [],
        "labels": [],
        "features": []
    }

    print("Starting feature extraction...")

    for i, genre_folder in enumerate(sorted(os.listdir(dataset_path))):

        genre_path = os.path.join(dataset_path, genre_folder)

        if os.path.isdir(genre_path):

            data["mapping"].append(genre_folder)
            print(f"\nProcessing genre: {genre_folder}")

            for filename in sorted(os.listdir(genre_path)):

                if filename.endswith(".wav"):

                    file_path = os.path.join(genre_path, filename)

                    try:
                        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                        if len(signal) >= SAMPLES_PER_TRACK:

                            num_samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)

                            for s in range(NUM_SEGMENTS):

                                start_sample = s * num_samples_per_segment
                                end_sample = start_sample + num_samples_per_segment
                                segment = signal[start_sample:end_sample]

                                # Feature extraction
                                mfccs = np.mean(librosa.feature.mfcc(
                                    y=segment, sr=sr, n_mfcc=NUM_MFCC,
                                    n_fft=N_FFT, hop_length=HOP_LENGTH
                                ), axis=1)

                                chroma = np.mean(librosa.feature.chroma_stft(
                                    y=segment, sr=sr,
                                    n_fft=N_FFT, hop_length=HOP_LENGTH
                                ), axis=1)

                                spectral_centroid = np.mean(
                                    librosa.feature.spectral_centroid(
                                        y=segment, sr=sr,
                                        n_fft=N_FFT, hop_length=HOP_LENGTH
                                    )
                                )

                                spectral_rolloff = np.mean(
                                    librosa.feature.spectral_rolloff(
                                        y=segment, sr=sr,
                                        n_fft=N_FFT, hop_length=HOP_LENGTH
                                    )
                                )

                                zcr = np.mean(
                                    librosa.feature.zero_crossing_rate(
                                        y=segment, hop_length=HOP_LENGTH
                                    )
                                )

                                # Combine features
                                feature_vector = np.hstack((
                                    mfccs, chroma,
                                    spectral_centroid,
                                    spectral_rolloff,
                                    zcr
                                ))

                                data["features"].append(feature_vector.tolist())
                                data["labels"].append(i)

                    except Exception as e:
                        print(f"Skipping {file_path}: {e}")
                        continue

    # Convert to DataFrame
    print("\nConverting data to pandas DataFrame...")
    df = pd.DataFrame(data["features"])
    df["genre_label"] = data["labels"]

    print(df.head())

    # Save CSV (MAIN OUTPUT)
    print(f"\nSaving CSV to {csv_path}...")
    df.to_csv(csv_path, index=False)

    # Optional JSON saving
    if save_json:
        print(f"Saving JSON to {JSON_PATH}...")
        with open(JSON_PATH, "w") as fp:
            json.dump(data, fp, indent=4)

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    process_dataset(DATASET_PATH, CSV_PATH, save_json=False)