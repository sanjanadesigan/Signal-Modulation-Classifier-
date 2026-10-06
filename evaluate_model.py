import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# FILES
# ============================================================

DATASET = "dataset_digicomm_assn.csv"
MODEL_FILE = "modulation_random_forest.pkl"


# ============================================================
# FEATURES
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
    # LOAD DATASET AND MODEL
    # ========================================================

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    df = pd.read_csv(DATASET)
    model = joblib.load(MODEL_FILE)

    print(f"\nDataset shape: {df.shape}")


    # ========================================================
    # PREPARE INPUTS AND TARGET
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


    # ========================================================
    # PREDICTIONS
    # ========================================================

    predictions = model.predict(X_test)


    # ========================================================
    # OVERALL ACCURACY
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE")
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
    # CONFUSION MATRIX
    # ========================================================

    labels = [
        "BPSK",
        "BFSK",
        "QPSK",
        "DPSK",
        "8-QAM",
        "16-QAM"
    ]

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    print("\nConfusion Matrix:")
    print(cm)


    plt.figure(figsize=(9, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel("Predicted Modulation")
    plt.ylabel("Actual Modulation")

    plt.title(
        "Modulation Classification Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png",
        dpi=300
    )

    plt.show()


    # ========================================================
    # ACCURACY VS SNR
    # ========================================================

    test_indices = X_test.index

    test_results = df.loc[test_indices].copy()

    test_results["Prediction"] = predictions

    snr_accuracy = (
        test_results
        .groupby("SNR_dB")
        .apply(
            lambda group:
            (
                group["Modulation"]
                == group["Prediction"]
            ).mean() * 100,
            include_groups=False
        )
    )

    print("\n" + "=" * 60)
    print("ACCURACY VS SNR")
    print("=" * 60)

    print(
        snr_accuracy.to_string()
    )


    # ========================================================
    # ACCURACY VS SNR PLOT
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        snr_accuracy.index,
        snr_accuracy.values,
        marker="o"
    )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Classification Accuracy (%)")

    plt.title(
        "Modulation Classification Accuracy vs SNR"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "accuracy_vs_snr.png",
        dpi=300
    )

    plt.show()


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    print(
        importance.to_string(
            index=False
        )
    )


    # ========================================================
    # FEATURE IMPORTANCE PLOT
    # ========================================================

    plt.figure(figsize=(10, 7))

    plt.barh(
        importance["Feature"],
        importance["Importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        "feature_importance.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":
    main()  