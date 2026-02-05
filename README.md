# A quantum computing framework for VLBI data correlation
## Introduction
We present a quantum computing framework for VLBI data correlation. We point out that a classical baseband time series data of length $N$ can be embedded into a quantum superposition state using amplitude encoding with only $\log_2 N$ qubits. The basic VLBI correlation and fringe fitting operations, including fringe rotation, Fourier transform, delay compensation, and cross correlation, can be implemented via quantum algorithms with significantly reduced computational complexity. We construct a full quantum processing pipeline and validate its feasibility and accuracy through direct comparison with a classical VLBI pipeline. We recognize that amplitude encoding of large data volumes remains the primary bottleneck in quantum computing; however, the quantized nature of VLBI raw data helps reduce the state-preparation complexity. Our investigation demonstrates that quantum computation offers a promising paradigm for VLBI data correlation and is likely to play a role in future VLBI systems.

**Note:**
If you make use of the quantum pipeline in this repo, please quote the repo link: `https://github.com/liulei/qcorr` and cite the paper:

- `Lei Liu, "A quantum computing framework for VLBI data correlation", 2026, arXiv:2602.04269`