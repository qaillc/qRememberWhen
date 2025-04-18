import streamlit as st
import subprocess, sys, os

# 1) Compute the absolute path to qml_platform.py
HERE = os.path.dirname(os.path.abspath(__file__))
QML_SCRIPT = os.path.join(HERE, "qml_platform.py")

st.title("QubitPred: QML on Qiskit 2")
st.write("Hybrid Variational Quantum Classifier using Aer simulator")

data_file = st.file_uploader("Upload CSV (last column label)", type=["csv"])
test_size = st.slider("Test size proportion", 0.1, 0.5, 0.2, step=0.05)
reps = st.slider("Repetitions (feature map & ansatz)", 1, 4, 2)
seed = st.number_input("Random seed", min_value=0, value=42)

if st.button("Run QML Pipeline"):
    # 2) Build the exact python call
    args = [
        sys.executable,
        QML_SCRIPT,
        "--test-size", str(test_size),
        "--reps",      str(reps),
        "--seed",      str(seed)
    ]

    # 3) If a CSV was uploaded, dump it next to the scripts
    if data_file:
        temp_csv = os.path.join(HERE, "temp.csv")
        with open(temp_csv, "wb") as f:
            f.write(data_file.getbuffer())
        args += ["--data", temp_csv]

    # 4) Run from the same folder as your scripts
    with st.spinner("Running training and evaluation…"):
        output = subprocess.check_output(args, cwd=HERE, stderr=subprocess.STDOUT)
    st.text(output.decode())
