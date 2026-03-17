import streamlit as st
import numpy as np

st.title("Hidden Markov Model (HMM) - All 3 Algorithms")

states = ['Rainy', 'Sunny']
obs_seq = ['U', 'U', 'N']

# Initial parameters
pi = np.array([0.6, 0.4])

A = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

B = np.array([
    [0.9, 0.1],  # Rainy: U, N
    [0.2, 0.8]   # Sunny: U, N
])

obs_map = {'U': 0, 'N': 1}

# ---------------- FORWARD ----------------
def forward(obs_seq, pi, A, B):
    T = len(obs_seq)
    N = len(pi)
    alpha = np.zeros((T, N))

    # Initialization
    for i in range(N):
        alpha[0][i] = pi[i] * B[i][obs_map[obs_seq[0]]]

    # Recursion
    for t in range(1, T):
        for j in range(N):
            alpha[t][j] = sum(alpha[t-1][i] * A[i][j] for i in range(N)) * B[j][obs_map[obs_seq[t]]]

    return alpha, np.sum(alpha[-1])


# ---------------- VITERBI ----------------
def viterbi(obs_seq, pi, A, B):
    T = len(obs_seq)
    N = len(pi)

    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    # Initialization
    for i in range(N):
        delta[0][i] = pi[i] * B[i][obs_map[obs_seq[0]]]

    # Recursion
    for t in range(1, T):
        for j in range(N):
            probs = [delta[t-1][i] * A[i][j] for i in range(N)]
            psi[t][j] = np.argmax(probs)
            delta[t][j] = max(probs) * B[j][obs_map[obs_seq[t]]]

    # Backtracking
    path = [np.argmax(delta[T-1])]
    for t in range(T-1, 0, -1):
        path.insert(0, psi[t][path[0]])

    state_path = [states[i] for i in path]
    return state_path, max(delta[T-1])


# ---------------- BAUM-WELCH ----------------
def baum_welch(obs_seq, pi, A, B, iterations=5):
    T = len(obs_seq)
    N = len(pi)

    for _ in range(iterations):
        # Forward
        alpha, prob = forward(obs_seq, pi, A, B)

        # Backward
        beta = np.zeros((T, N))
        beta[-1] = np.ones(N)

        for t in range(T-2, -1, -1):
            for i in range(N):
                beta[t][i] = sum(A[i][j] * B[j][obs_map[obs_seq[t+1]]] * beta[t+1][j] for j in range(N))

        # Gamma and Xi
        gamma = np.zeros((T, N))
        xi = np.zeros((T-1, N, N))

        for t in range(T):
            denom = sum(alpha[t][i] * beta[t][i] for i in range(N))
            for i in range(N):
                gamma[t][i] = (alpha[t][i] * beta[t][i]) / denom

        for t in range(T-1):
            denom = sum(alpha[t][i] * A[i][j] * B[j][obs_map[obs_seq[t+1]]] * beta[t+1][j]
                        for i in range(N) for j in range(N))
            for i in range(N):
                for j in range(N):
                    xi[t][i][j] = (alpha[t][i] * A[i][j] * B[j][obs_map[obs_seq[t+1]]] * beta[t+1][j]) / denom

        # Update pi
        pi = gamma[0]

        # Update A
        for i in range(N):
            for j in range(N):
                A[i][j] = sum(xi[t][i][j] for t in range(T-1)) / sum(gamma[t][i] for t in range(T-1))

        # Update B
        for j in range(N):
            for k in range(2):  # U, N
                B[j][k] = sum(gamma[t][j] for t in range(T) if obs_map[obs_seq[t]] == k) / sum(gamma[t][j] for t in range(T))

    return pi, A, B


# ---------------- UI ----------------
st.write("Observation Sequence:", obs_seq)

if st.button("Run Forward Algorithm"):
    alpha, prob = forward(obs_seq, pi, A, B)
    st.success(f"Probability: {prob}")
    st.write("Alpha Table:")
    st.write(alpha)

if st.button("Run Viterbi Algorithm"):
    path, prob = viterbi(obs_seq, pi, A, B)
    st.success(f"Best Path: {path}")
    st.write(f"Probability: {prob}")

if st.button("Run Baum-Welch Learning"):
    new_pi, new_A, new_B = baum_welch(obs_seq, pi.copy(), A.copy(), B.copy())

    st.success("Updated Parameters after Learning")
    st.write("Initial Probabilities:", new_pi)
    st.write("Transition Matrix:", new_A)
    st.write("Emission Matrix:", new_B)