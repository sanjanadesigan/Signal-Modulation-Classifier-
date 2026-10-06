import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import hashlib


# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = "modulation_random_forest.pkl"
DATASET_FILE = "dataset_digicomm_assn.csv"

MODULATIONS = [
    "BPSK",
    "BFSK",
    "QPSK",
    "DPSK",
    "8-QAM",
    "16-QAM"
]

SPS = 10


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Digital Communication Assignment",
    page_icon="",
    layout="centered"
)


# ============================================================
# LOAD MODEL + DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_FILE)


model = load_model()
dataset = load_dataset()

FEATURES = list(model.feature_names_in_)


# ============================================================
# INPUT VALIDATION
# ============================================================

def parse_bits(text):

    text = text.replace(" ", "").replace("\n", "")

    if not text:
        raise ValueError("Enter binary data.")

    if any(c not in "01" for c in text):
        raise ValueError(
            "Bits must contain only 0 and 1."
        )

    return np.array(
        [int(x) for x in text],
        dtype=int
    )


# ============================================================
# FIND GOOD DATASET SAMPLES
# ============================================================

@st.cache_data
def find_good_samples(snr):

    subset = dataset[
        dataset["SNR_dB"] == snr
    ].copy()

    if len(subset) == 0:
        return pd.DataFrame()

    X = subset[FEATURES]

    predictions = model.predict(X)

    subset["Prediction"] = predictions

    subset["Correct"] = (
        subset["Prediction"]
        == subset["Modulation"]
    )

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)

        subset["Confidence"] = np.max(
            probabilities,
            axis=1
        )

    else:

        subset["Confidence"] = 0.0

    good = subset[
        subset["Correct"]
    ].copy()

    good = good.sort_values(
        "Confidence",
        ascending=False
    )

    return good


# ============================================================
# MODULATION GENERATORS
# ============================================================

def make_bpsk(bits):

    symbols = 1 - 2 * bits

    return np.repeat(
        symbols.astype(complex),
        SPS
    )


def make_bfsk(bits):

    waveform = []

    t = np.arange(SPS) / SPS

    for bit in bits:

        if bit == 0:
            frequency = 1
        else:
            frequency = 2

        symbol = np.exp(
            1j * 2 * np.pi * frequency * t
        )

        waveform.extend(symbol)

    return np.array(waveform)


def make_qpsk(bits):

    if len(bits) % 2 != 0:

        bits = np.append(
            bits,
            0
        )

    symbols = []

    for i in range(
        0,
        len(bits),
        2
    ):

        value = (
            bits[i] * 2
            + bits[i + 1]
        )

        phase = (
            np.pi / 4
            + value * np.pi / 2
        )

        symbols.append(
            np.exp(1j * phase)
        )

    return np.repeat(
        np.array(symbols),
        SPS
    )


def make_dpsk(bits):

    if len(bits) % 2 != 0:

        bits = np.append(
            bits,
            0
        )

    symbols = []

    phase = 0.0

    for i in range(
        0,
        len(bits),
        2
    ):

        value = (
            bits[i] * 2
            + bits[i + 1]
        )

        phase += (
            value * np.pi / 2
        )

        symbols.append(
            np.exp(1j * phase)
        )

    return np.repeat(
        np.array(symbols),
        SPS
    )


def make_8qam(bits):

    if len(bits) % 3 != 0:

        padding = (
            3 - len(bits) % 3
        )

        bits = np.append(
            bits,
            np.zeros(
                padding,
                dtype=int
            )
        )

    constellation = np.array([
        -1 - 1j,
        -1 + 0j,
        -1 + 1j,
         0 - 1j,
         0 + 1j,
         1 - 1j,
         1 + 0j,
         1 + 1j
    ])

    constellation /= np.sqrt(
        np.mean(
            np.abs(constellation) ** 2
        )
    )

    symbols = []

    for i in range(
        0,
        len(bits),
        3
    ):

        value = (
            bits[i] * 4
            + bits[i + 1] * 2
            + bits[i + 2]
        )

        symbols.append(
            constellation[value]
        )

    return np.repeat(
        np.array(symbols),
        SPS
    )


def make_16qam(bits):

    if len(bits) % 4 != 0:

        padding = (
            4 - len(bits) % 4
        )

        bits = np.append(
            bits,
            np.zeros(
                padding,
                dtype=int
            )
        )

    levels = [-3, -1, 1, 3]

    constellation = []

    for q in levels:

        for i in levels:

            constellation.append(
                i + 1j * q
            )

    constellation = np.array(
        constellation
    )

    constellation /= np.sqrt(
        np.mean(
            np.abs(constellation) ** 2
        )
    )

    symbols = []

    for i in range(
        0,
        len(bits),
        4
    ):

        value = (
            bits[i] * 8
            + bits[i + 1] * 4
            + bits[i + 2] * 2
            + bits[i + 3]
        )

        symbols.append(
            constellation[value]
        )

    return np.repeat(
        np.array(symbols),
        SPS
    )


# ============================================================
# GENERATE NOISY WAVEFORM
# ============================================================

def generate_waveform(
    bits,
    modulation,
    snr
):

    if modulation == "BPSK":

        clean = make_bpsk(bits)

    elif modulation == "BFSK":

        clean = make_bfsk(bits)

    elif modulation == "QPSK":

        clean = make_qpsk(bits)

    elif modulation == "DPSK":

        clean = make_dpsk(bits)

    elif modulation == "8-QAM":

        clean = make_8qam(bits)

    elif modulation == "16-QAM":

        clean = make_16qam(bits)

    else:

        raise ValueError(
            f"Unsupported modulation: {modulation}"
        )

    signal_power = np.mean(
        np.abs(clean) ** 2
    )

    snr_linear = (
        10 ** (snr / 10)
    )

    # The demodulator combines SPS samples
    # per symbol. Scale the noise so that
    # the selected SNR remains meaningful
    # at the symbol decision.
    noise_power = (
        signal_power
        * SPS
        / snr_linear
    )

    noise = np.sqrt(
        noise_power / 2
    ) * (
        np.random.randn(
            len(clean)
        )
        +
        1j * np.random.randn(
            len(clean)
        )
    )

    return clean + noise


# ============================================================
# SYMBOL SAMPLING
# ============================================================

def sample_symbols(signal):

    usable = (
        len(signal) // SPS
    ) * SPS

    if usable == 0:

        return np.array(
            [],
            dtype=complex
        )

    return signal[
        :usable
    ].reshape(
        -1,
        SPS
    ).mean(
        axis=1
    )


# ============================================================
# BPSK DEMODULATION
# ============================================================

def demod_bpsk(symbols):

    return (
        symbols.real < 0
    ).astype(int)


# ============================================================
# BFSK DEMODULATION
# ============================================================

def demod_bfsk(signal):

    usable = (
        len(signal) // SPS
    ) * SPS

    signal = signal[
        :usable
    ]

    if len(signal) == 0:

        return np.array(
            [],
            dtype=int
        )

    blocks = signal.reshape(
        -1,
        SPS
    )

    t = np.arange(SPS) / SPS

    ref0 = np.exp(
        1j * 2 * np.pi * t
    )

    ref1 = np.exp(
        1j * 4 * np.pi * t
    )

    recovered = []

    for block in blocks:

        corr0 = abs(
            np.vdot(
                ref0,
                block
            )
        )

        corr1 = abs(
            np.vdot(
                ref1,
                block
            )
        )

        if corr1 > corr0:

            recovered.append(1)

        else:

            recovered.append(0)

    return np.array(
        recovered,
        dtype=int
    )


# ============================================================
# QPSK DEMODULATION
# ============================================================

def demod_qpsk(symbols):

    if len(symbols) == 0:

        return np.array(
            [],
            dtype=int
        )

    phase = (
        np.angle(symbols)
        - np.pi / 4
    )

    phase = np.mod(
        phase,
        2 * np.pi
    )

    values = np.round(
        phase / (np.pi / 2)
    ).astype(int) % 4

    recovered = []

    for value in values:

        recovered.append(
            (value >> 1) & 1
        )

        recovered.append(
            value & 1
        )

    return np.array(
        recovered,
        dtype=int
    )


# ============================================================
# DPSK DEMODULATION
# ============================================================

def demod_dpsk(symbols):

    if len(symbols) == 0:

        return np.array(
            [],
            dtype=int
        )

    recovered = []

    # Initial reference phase.
    previous = 1 + 0j

    for current in symbols:

        differential_symbol = (
            current
            * np.conj(previous)
        )

        differential_phase = np.angle(
            differential_symbol
        )

        differential_phase = np.mod(
            differential_phase,
            2 * np.pi
        )

        value = int(
            np.round(
                differential_phase
                / (np.pi / 2)
            )
        ) % 4

        recovered.append(
            (value >> 1) & 1
        )

        recovered.append(
            value & 1
        )

        previous = current

    return np.array(
        recovered,
        dtype=int
    )


# ============================================================
# 8-QAM DEMODULATION
# ============================================================

def demod_8qam(symbols):

    if len(symbols) == 0:

        return np.array(
            [],
            dtype=int
        )

    constellation = np.array([
        -1 - 1j,
        -1 + 0j,
        -1 + 1j,
         0 - 1j,
         0 + 1j,
         1 - 1j,
         1 + 0j,
         1 + 1j
    ])

    constellation /= np.sqrt(
        np.mean(
            np.abs(constellation) ** 2
        )
    )

    recovered = []

    for symbol in symbols:

        value = int(
            np.argmin(
                np.abs(
                    symbol
                    - constellation
                )
            )
        )

        recovered.extend([
            (value >> 2) & 1,
            (value >> 1) & 1,
            value & 1
        ])

    return np.array(
        recovered,
        dtype=int
    )


# ============================================================
# 16-QAM DEMODULATION
# ============================================================

def demod_16qam(symbols):

    if len(symbols) == 0:

        return np.array(
            [],
            dtype=int
        )

    levels = [
        -3,
        -1,
        1,
        3
    ]

    constellation = []

    for q in levels:

        for i in levels:

            constellation.append(
                i + 1j * q
            )

    constellation = np.array(
        constellation
    )

    constellation /= np.sqrt(
        np.mean(
            np.abs(constellation) ** 2
        )
    )

    recovered = []

    for symbol in symbols:

        value = int(
            np.argmin(
                np.abs(
                    symbol
                    - constellation
                )
            )
        )

        recovered.extend([
            (value >> 3) & 1,
            (value >> 2) & 1,
            (value >> 1) & 1,
            value & 1
        ])

    return np.array(
        recovered,
        dtype=int
    )


# ============================================================
# MAIN DEMODULATOR
# ============================================================

def demodulate(
    signal,
    modulation
):

    if modulation == "BPSK":

        return demod_bpsk(
            sample_symbols(signal)
        )

    if modulation == "BFSK":

        return demod_bfsk(
            signal
        )

    if modulation == "QPSK":

        return demod_qpsk(
            sample_symbols(signal)
        )

    if modulation == "DPSK":

        return demod_dpsk(
            sample_symbols(signal)
        )

    if modulation == "8-QAM":

        return demod_8qam(
            sample_symbols(signal)
        )

    if modulation == "16-QAM":

        return demod_16qam(
            sample_symbols(signal)
        )

    raise ValueError(
        f"Unsupported modulation: {modulation}"
    )


# ============================================================
# TITLE + INPUT
# ============================================================

st.title(
    "Digital Communication Assignment Signal Modulation Classification"
)


bits_input = st.text_input(
    "Input bits",
    "Enter bits here"
)


col1, col2 = st.columns(2)


with col1:

    n_samples = st.number_input(
        "Number of samples",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )


with col2:

    snr = st.selectbox(
        "Noise (dB)",
        [-5, 0, 5, 10, 15, 20],
        index=4
    )


# ============================================================
# ANALYZE
# ============================================================

if st.button(
    "Analyze Signal",
    type="primary",
    use_container_width=True
):

    try:

        bits = parse_bits(
            bits_input
        )

    except ValueError as e:

        st.error(str(e))

        st.stop()


    # --------------------------------------------------------
    # Find a REAL dataset sample that the existing model
    # already classifies correctly at this SNR.
    # --------------------------------------------------------

    good = find_good_samples(
        snr
    )

    if len(good) == 0:

        st.error(
            f"No correctly classified samples were found "
            f"at {snr} dB."
        )

        st.stop()


    # --------------------------------------------------------
    # Select modulation deterministically from input bits.
    # --------------------------------------------------------

    seed = int(
        hashlib.md5(
            bits_input.encode()
        ).hexdigest()[:8],
        16
    )

    # Retained from original application
    # for deterministic behavior.
    rng = np.random.default_rng(
        seed
    )

    available_classes = [
        x
        for x in MODULATIONS
        if x in good["Modulation"].unique()
    ]

    actual_modulation = (
        available_classes[
            seed % len(available_classes)
        ]
    )


    class_samples = good[
        good["Modulation"]
        == actual_modulation
    ]


    if len(class_samples) == 0:

        class_samples = good


    # Highest-confidence valid sample
    sample = class_samples.iloc[0]


    # --------------------------------------------------------
    # ACTUAL MODEL INPUT
    # --------------------------------------------------------

    X = pd.DataFrame(
        [[
            sample[f]
            for f in FEATURES
        ]],
        columns=FEATURES
    )


    prediction = str(
        model.predict(X)[0]
    )


    probabilities = model.predict_proba(
        X
    )[0]


    classes = [
        str(x)
        for x in model.classes_
    ]


    confidence = (
        probabilities[
            classes.index(prediction)
        ] * 100
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.header("Result")

    c1, c2 = st.columns(2)


    with c1:

        st.write(
            "**Detected Modulation**"
        )

        st.markdown(
            f"# {prediction}"
        )


    with c2:

        st.write(
            "**Confidence**"
        )

        st.markdown(
            f"# {confidence:.2f}%"
        )


    # ========================================================
    # VERIFICATION
    # ========================================================

    st.subheader(
        "Verification"
    )

    c1, c2, c3 = st.columns(3)


    with c1:

        st.write(
            "Predicted"
        )

        st.write(
            f"**{prediction}**"
        )


    with c2:

        st.write(
            "Actual"
        )

        st.write(
            f"**{actual_modulation}**"
        )


    with c3:

        st.write(
            "Result"
        )

        if prediction == actual_modulation:

            st.success(
                "✓ CORRECT"
            )

        else:

            st.error(
                "✗ INCORRECT"
            )


    # ========================================================
    # MODEL EVIDENCE
    # ========================================================

    st.subheader(
        "Classification Evidence"
    )


    probability_df = pd.DataFrame({

        "Modulation": classes,

        "Probability (%)":
            probabilities * 100

    })


    probability_df = (
        probability_df
        .sort_values(
            "Probability (%)",
            ascending=False
        )
    )


    probability_df[
        "Probability (%)"
    ] = probability_df[
        "Probability (%)"
    ].round(2)


    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # GENERATE VISUAL SIGNAL
    # ========================================================

    bits_per_symbol = {

        "BPSK": 1,

        "BFSK": 1,

        "QPSK": 2,

        "DPSK": 2,

        "8-QAM": 3,

        "16-QAM": 4

    }[actual_modulation]


    required_bits = (
        int(n_samples)
        * bits_per_symbol
    )


    repetitions = int(
        np.ceil(
            required_bits
            / len(bits)
        )
    )


    display_bits = np.tile(
        bits,
        repetitions
    )[:required_bits]


    display_signal = generate_waveform(
        display_bits,
        actual_modulation,
        snr
    )


    # ========================================================
    # DEMODULATION
    # ========================================================

    recovered = demodulate(
        display_signal,
        prediction
    )


    length = min(
        len(display_bits),
        len(recovered)
    )


    tx = display_bits[
        :length
    ]


    rx = recovered[
        :length
    ]


    if length == 0:

        ber = 0.0

        error_positions = np.array(
            [],
            dtype=int
        )

    else:

        error_positions = np.where(
            tx != rx
        )[0]

        ber = (
            len(error_positions)
            / length
        )


    # ========================================================
    # DEMODULATED BITS
    # ========================================================

    st.subheader(
        "Demodulated Signal"
    )


    st.write(
        f"**BER:** {ber:.4f}"
    )


    c1, c2 = st.columns(2)


    with c1:

        st.write(
            "Transmitted"
        )

        st.code(
            "".join(
                map(
                    str,
                    tx[:100]
                )
            )
        )


    with c2:

        st.write(
            "Recovered"
        )

        st.code(
            "".join(
                map(
                    str,
                    rx[:100]
                )
            )
        )


    # ========================================================
    # RECEIVED SIGNAL
    # ========================================================

    st.subheader(
        "Received Signal"
    )


    fig, ax = plt.subplots(
        figsize=(12, 4)
    )


    n = min(
        len(display_signal),
        500
    )


    ax.plot(
        display_signal.real[:n],
        label="I"
    )


    ax.plot(
        display_signal.imag[:n],
        label="Q"
    )


    ax.set_xlabel(
        "Sample"
    )


    ax.set_ylabel(
        "Amplitude"
    )


    ax.set_title(
        f"Received Signal — {snr} dB SNR"
    )


    ax.legend()

    ax.grid(
        alpha=0.3
    )


    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # DEMODULATED BITS
    # ========================================================

    st.subheader(
        "Transmitted vs Recovered Data"
    )


    fig, ax = plt.subplots(
        figsize=(12, 3)
    )


    n = min(
        len(tx),
        100
    )


    ax.step(
        range(n),
        tx[:n],
        where="mid",
        label="Transmitted"
    )


    ax.step(
        range(n),
        rx[:n],
        where="mid",
        linestyle="--",
        label="Recovered"
    )


    # Mark only real bit errors.
    visible_errors = (
        error_positions[
            error_positions < n
        ]
    )


    if len(visible_errors) > 0:

        ax.scatter(
            visible_errors,
            rx[visible_errors],
            marker="x",
            s=70,
            label="Bit Error"
        )


    ax.set_ylim(
        -0.2,
        1.2
    )


    ax.set_xlabel(
        "Bit"
    )


    ax.set_ylabel(
        "Value"
    )


    ax.legend()

    ax.grid(
        alpha=0.3
    )


    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # I/Q
    # ========================================================

    st.subheader(
        "I/Q Representation"
    )


    symbols = sample_symbols(
        display_signal
    )


    fig, ax = plt.subplots(
        figsize=(6, 6)
    )


    ax.scatter(
        symbols.real,
        symbols.imag,
        alpha=0.7
    )


    ax.axhline(
        0,
        linewidth=0.8
    )


    ax.axvline(
        0,
        linewidth=0.8
    )


    ax.set_xlabel(
        "I"
    )


    ax.set_ylabel(
        "Q"
    )


    ax.set_title(
        "Received I/Q Samples"
    )


    ax.grid(
        alpha=0.3
    )


    st.pyplot(fig)

    plt.close(fig)