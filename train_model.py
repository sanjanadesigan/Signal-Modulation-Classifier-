import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# FILES
# ============================================================

DATASET = "dataset_digicomm_assn.csv"
MODEL_FILE = "modulation_random_forest.pkl"


# ============================================================
# FEATURES
# ============================================================

# These are the features actually given to the ML model.
#
# Notice that:
#
# Preamble_BPSK_Corr
# Preamble_DPSK_Corr
#
# are intentionally NOT included.
#
# SNR_dB is also NOT included because the classifier should
# identify modulation without being told the SNR.
# ============================================================

FEATURE_COLUMNS = [

    "I_mean",
    "I_std",

    "Q_mean",
    "Q_std",

    "Amplitude_mean",
    "Amplitude_std",
    "Amplitude_max",
    "Amplitude_min",

    "Power",

    "Phase_std",
    "PhaseDiff_std",
    "PhaseDiff_abs_mean",

    "PreamblePhaseDiff_std",
    "PreamblePhaseDiff_abs_mean",

    "SpectralCentroid",
    "SpectralBandwidth",
    "SpectralEntropy"
]


def main():

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    df = pd.read_csv(DATASET)

    print(f"\nDataset shape: {df.shape}")

    print("\nClass distribution:")
    print(df["Modulation"].value_counts().sort_index())

    print("\nSNR distribution:")
    print(df["SNR_dB"].value_counts().sort_index())


    # ========================================================
    # INPUTS AND TARGET
    # ========================================================

    X = df[FEATURE_COLUMNS]
    y = df["Modulation"]


    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)


    # ========================================================
    # PREDICTION
    # ========================================================

    predictions = model.predict(X_test)


    # ========================================================
    # ACCURACY
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(
        f"\nOverall Accuracy: "
        f"{accuracy * 100:.2f}%"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            digits=4
        )
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"\nModel saved successfully as: "
        f"{MODEL_FILE}"
    )


if __name__ == "__main__":
    main()