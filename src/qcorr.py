#!/usr/bin/env python

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.synthesis import synth_qft_full as QFT
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

import utils

def normalize(vec):
    vec = np.asarray(vec, dtype=complex)
    return vec / np.linalg.norm(vec)


def build_linear_phase_gate(alpha, n_qubits):
    qc = QuantumCircuit(n_qubits)
    for j in range(n_qubits):
        theta_j = alpha * (2 ** j)
        qc.p(theta_j, j)
    return qc.to_gate(label=f"Phase(a={alpha:.3f})")

def build_theta_gate(n_qubits, theta):
    qc = QuantumCircuit(n_qubits)
    qc.p(theta, n_qubits-1)
    return qc.to_gate(label="LH")

def build_signal_pipeline_gate(time_samples,
                               alpha_fr,
                               alpha_tau,
                               n_qubits=3,
                               theta=0.0,
                               label="Pipe"):
    # 1. amplitude encoding
    amps = normalize(time_samples)
#    init = Initialize(amps)
#    init.label = "Init"
    init    =   StatePreparation(amps, label='Init')

    # 2. fringe rotation
    D_time = build_linear_phase_gate(alpha_fr, n_qubits)

    # 3. QFT
    qft = QFT(num_qubits=n_qubits, do_swaps=True, inverse=True)
    qft_gate = qft.to_gate(label="QFT")

    # 4. FSTC
    D_freq = build_linear_phase_gate(alpha_tau, n_qubits)

    # sub circuit
    sub = QuantumCircuit(n_qubits)
    sub.append(init, range(n_qubits))      # Init |000> -> |psi_t>
    sub.append(D_time, range(n_qubits))    # Fringe rotation for delay model
    sub.append(qft_gate, range(n_qubits))  # QFT
    sub.append(D_freq, range(n_qubits))    # FSTC for delay model

    # 5. phase gate to select L and H part freqs
    if theta != 0.0:
        qc_LH = QuantumCircuit(n_qubits)
        qc_LH.p(theta, n_qubits-1)
        qc_LH.to_gate(label="LH")
        sub.append(qc_LH, range(n_qubits))    # Fringe fitting

    return sub.to_gate(label=label)

def build_corr_gate(UA, UB, n_qubits, theta=0.0, alpha=0.0):

    sub_corr = QuantumCircuit(n_qubits)
    sub_corr.append(UA, range(n_qubits))

    tau_gate    =   build_linear_phase_gate(alpha, n_qubits)
    sub_corr.append(tau_gate, range(n_qubits))

    lh_gate =   build_theta_gate(n_qubits, theta)
    sub_corr.append(lh_gate, range(n_qubits))

    sub_corr.append(UB.inverse(), range(n_qubits))
    Ucorr = sub_corr.to_gate(label="U_corr")

    return Ucorr

def hadamard_test_F(UA, UB, n_qubits, theta=0.0, alpha=0.0, part="re", shots=20000):

    assert part in ['re', 'im']

    Ucorr = build_corr_gate(UA, UB, n_qubits, theta=theta, alpha=alpha)

    # 4 qubits：
    # 0: ancilla
    # 1, 2, 3: data (3-qubit for q0,q1,q2)
    qc = QuantumCircuit(1+n_qubits, 1)
    anc = 0
    data = list(range(1, n_qubits+1))

    # Hadamard test
    qc.h(anc)
    if part == "im":
        # S^\dagger for Imag
        qc.sdg(anc)

    # controlled-U(\theta)
    qc.append(Ucorr.control(1), [anc] + data)

    qc.h(anc)
    qc.measure(anc, 0)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    p0 = counts.get('0', 0) / shots
    val = 2 * p0 - 1  # Re or Im

    return val, counts, qc

def main():
    n_qubits = 8
    N = 2 ** n_qubits
    BW  =   0.1E6
    Fs  =   BW * 2
    Ts  =   1. / Fs
    df_fr   =   500.0
    tau_frac    =   4E-6
    f_res   =   Fs / N
    tau_res =   2.7E-6

    buf1, buf2, cfg  =   utils.gen_raw(N, BW, df_fr, tau_frac, tau_res)

    alpha_fr    =   2 * np.pi * df_fr * Ts
    alpha_tau   =   2 * np.pi * tau_frac * f_res

    # 3. build corr pipeline gate:
    UA = build_signal_pipeline_gate(buf1,
                                    alpha_fr=-alpha_fr,
                                    alpha_tau=-alpha_tau,
                                    n_qubits=n_qubits,
                                    label="UA")
    UB = build_signal_pipeline_gate(buf2,
                                    alpha_fr=0.0,
                                    alpha_tau=0.0,
                                    n_qubits=n_qubits,
                                    label="UB")

    taus    =   np.arange(-10, 10+1, 1) * 1E-6
#    vsums_c  =   utils.do_corr_classic(cfg, buf1, buf2, taus)
#    utils.do_fit(cfg, taus*1E6, vsums_c, name='classic')

    vsums   =   []
    for tau in taus:
        alpha   =   -2 * np.pi * tau * f_res
        vsum    =   do_corr_fit(cfg, n_qubits, UA, UB, alpha) 
        print('qc: tau = %f us, amp: %f' %  (tau * 1E6, np.abs(vsum))) 
        vsums.append(vsum)
    utils.do_fit(cfg, taus*1E6, vsums, name='quantum')

#    np.save('tau_c_q.npy', [taus, vsums_c, vsums])

def do_corr_fit(cfg, n_qubits, UA, UB, alpha):

#    alpha   =   alpha_tau * 0.5
#    alpha   =   0.0

    # 1) Hadamard test for F(0) = <000| U(0) |000>
    Re_F0, counts_re0, _ = hadamard_test_F(UA, UB, n_qubits, theta=0.0, alpha=alpha, part="re")
    Im_F0, counts_im0, _ = hadamard_test_F(UA, UB, n_qubits, theta=0.0, alpha=alpha, part="im")

    # 2) Hadamard test for F(\pi) = <000| U(\pi) |000>
    Re_Fpi, counts_repi, _ = hadamard_test_F(UA, UB, n_qubits, theta=np.pi, alpha=alpha, part="re")
    Im_Fpi, counts_impi, _ = hadamard_test_F(UA, UB, n_qubits, theta=np.pi, alpha=alpha, part="im")

    # 3) F(0)=L+H, F(\pi)=L-H
    Re_H_est = (Re_F0 - Re_Fpi) / 2
    Im_H_est = (Im_F0 - Im_Fpi) / 2

    Re_L_est = (Re_F0 + Re_Fpi) / 2
    Im_L_est = (Im_F0 + Im_Fpi) / 2

    vsum    =   Re_L_est + 1j*Im_L_est
    return vsum

    print('L_est: %.4f + %.4fj, angle %.4f' % (Re_L_est, Im_L_est, np.angle(Re_L_est + 1j*Im_L_est)))
#    print('H_est: %.4f + %.4fj, angle %.4f' % (Re_H_est, Im_H_est, np.angle(Re_H_est + 1j*Im_H_est)))

# append delay search tau 
    qc  =   QuantumCircuit(n_qubits)
    qc.append(UA, range(n_qubits))
    D_freq = build_linear_phase_gate(alpha, n_qubits)
    qc.append(D_freq, range(n_qubits))
    UA  =   qc.to_gate(label='qc_tau')
    
    Nh  =   N // 2
    sv0 = Statevector.from_label("0" * n_qubits)
    psiA_final = sv0.evolve(UA)
    psiB_final = sv0.evolve(UB)
    overlap_L_true = np.vdot(psiB_final.data[:Nh], psiA_final.data[:Nh])  # <psi_B_final|psi_A_final>
    Re_true = np.real(overlap_L_true)
    Im_true = np.imag(overlap_L_true)

    print('L_true: ', overlap_L_true, 'angle: ', np.angle(overlap_L_true))
    print()

if __name__ == "__main__":
    main()
