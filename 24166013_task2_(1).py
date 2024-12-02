import numpy as np

def steady_state_finder(trans_matrix, iterations=1000, tolerance=1e-6):
    num_states = trans_matrix.shape[0]
    dist = np.full(num_states, 1 / num_states)
    for _ in range(iterations):
        updated_dist = np.dot(dist, trans_matrix)
        if np.max(np.abs(updated_dist - dist)) < tolerance:
            break
        dist = updated_dist
    return updated_dist
def fwd_algorithm(init_probs, emission_probs, trans_probs, obs_seq):
    time_steps, num_states = len(obs_seq), len(init_probs)
    alpha_vals = np.zeros((time_steps, num_states))
    alpha_vals[0] = init_probs * emission_probs[:, obs_seq[0]] + 1e-10
    for t in range(1, time_steps):
        for state in range(num_states):
            alpha_vals[t, state] = np.sum(alpha_vals[t - 1] * trans_probs[:, state]) * emission_probs[state, obs_seq[t]] + 1e-10
    return alpha_vals
def bwd_algorithm(emission_probs, trans_probs, obs_seq):
    time_steps, num_states = len(obs_seq), trans_probs.shape[0]
    beta_vals = np.zeros((time_steps, num_states))
    beta_vals[-1] = 1
    for t in reversed(range(time_steps - 1)):
        for state in range(num_states):
            beta_vals[t, state] = np.sum(trans_probs[state, :] * emission_probs[:, obs_seq[t + 1]] * beta_vals[t + 1]) + 1e-10
    return beta_vals
def baum_welch_algorithm(obs_seqs, trans_probs, emission_probs, init_probs, num_states, num_obs, max_iterations=200):
    epsilon = 1e-10
    for obs_seq in obs_seqs:
        seq_length = len(obs_seq)
        for _ in range(max_iterations):
            alpha_vals = fwd_algorithm(init_probs, emission_probs, trans_probs, obs_seq)
            beta_vals = bwd_algorithm(emission_probs, trans_probs, obs_seq)

            gamma_vals = np.zeros((seq_length, num_states))
            xi_vals = np.zeros((seq_length - 1, num_states, num_states))

            for t in range(seq_length):
                gamma_sum = np.sum(alpha_vals[t] * beta_vals[t]) + epsilon
                gamma_vals[t] = (alpha_vals[t] * beta_vals[t]) / gamma_sum

            for t in range(seq_length - 1):
                xi_sum = np.sum(np.outer(alpha_vals[t], beta_vals[t + 1]) * trans_probs * emission_probs[:, obs_seq[t + 1]]) + epsilon
                for i in range(num_states):
                    for j in range(num_states):
                        xi_vals[t, i, j] = (alpha_vals[t, i] * trans_probs[i, j] * emission_probs[j, obs_seq[t + 1]] * beta_vals[t + 1, j]) / xi_sum

            init_probs = gamma_vals[0] + epsilon

            for i in range(num_states):
                trans_probs[i] = np.sum(xi_vals[:, i, :], axis=0) / (np.sum(gamma_vals[:-1, i]) + epsilon)

            for i in range(num_states):
                for k in range(num_obs):
                    emission_sum = np.sum(gamma_vals[obs_seq == k, i])
                    state_sum = np.sum(gamma_vals[:, i]) + epsilon
                    emission_probs[i, k] = emission_sum / state_sum

    return trans_probs, emission_probs, init_probs

def encode_obs_seq(obs_seq, obs_labels):
    return np.array([obs_labels.index(obs) for obs in obs_seq])

def get_states_and_obs(state_labels, obs_labels):
    return len(state_labels), len(obs_labels)

if __name__ == "__main__":
    states, obs_labels = ['H', 'L'], ['R', 'D', 'E', 'F', 'G']
    num_states, num_obs = get_states_and_obs(states, obs_labels)

    obs_sequences = [
        ['R', 'R', 'D', 'D', 'R', 'D', 'R'],
        ['D', 'R', 'D', 'D', 'R', 'R', 'D'],
        ['R', 'D', 'D', 'D', 'R', 'D', 'R'],
        ['R', 'R', 'R', 'D', 'D', 'R', 'D'],
        ['D', 'R', 'D', 'D', 'D', 'D'import numpy as np

# Steady-state finder
def steady_state_finder(trans_matrix, iterations=1000, tolerance=1e-6):
    num_states = trans_matrix.shape[0]
    dist = np.full(num_states, 1 / num_states)
    for _ in range(iterations):
        updated_dist = np.dot(dist, trans_matrix)
        if np.max(np.abs(updated_dist - dist)) < tolerance:
            break
        dist = updated_dist
    return updated_dist

# Forward algorithm
def fwd_algorithm(init_probs, emission_probs, trans_probs, obs_seq):
    time_steps, num_states = len(obs_seq), len(init_probs)
    alpha_vals = np.zeros((time_steps, num_states))
    alpha_vals[0] = init_probs * emission_probs[:, obs_seq[0]] + 1e-10
    for t in range(1, time_steps):
        for state in range(num_states):
            alpha_vals[t, state] = np.sum(alpha_vals[t - 1] * trans_probs[:, state]) * emission_probs[state, obs_seq[t]] + 1e-10
    return alpha_vals

# Backward algorithm
def bwd_algorithm(emission_probs, trans_probs, obs_seq):
    time_steps, num_states = len(obs_seq), trans_probs.shape[0]
    beta_vals = np.zeros((time_steps, num_states))
    beta_vals[-1] = 1
    for t in reversed(range(time_steps - 1)):
        for state in range(num_states):
            beta_vals[t, state] = np.sum(trans_probs[state, :] * emission_probs[:, obs_seq[t + 1]] * beta_vals[t + 1]) + 1e-10
    return beta_vals

# Likelihood calculation
def calculate_likelihood(alpha_vals):
    return np.log(np.sum(alpha_vals[-1]) + 1e-10)

# Baum-Welch algorithm with memory mechanism
def baum_welch_algorithm_with_memory(obs_seqs, trans_probs, emission_probs, init_probs, num_states, num_obs, max_iterations=200, tolerance=1e-4):
    epsilon = 1e-10

    # Store the best parameters and likelihood
    best_trans_probs = trans_probs.copy()
    best_emission_probs = emission_probs.copy()
    best_init_probs = init_probs.copy()
    best_likelihood = -np.inf

    for obs_seq in obs_seqs:
        seq_length = len(obs_seq)
        for iteration in range(max_iterations):
            alpha_vals = fwd_algorithm(init_probs, emission_probs, trans_probs, obs_seq)
            beta_vals = bwd_algorithm(emission_probs, trans_probs, obs_seq)

            # Calculate current likelihood
            current_likelihood = calculate_likelihood(alpha_vals)

            # Update best parameters if the likelihood improves
            if current_likelihood > best_likelihood:
                best_likelihood = current_likelihood
                best_trans_probs = trans_probs.copy()
                best_emission_probs = emission_probs.copy()
                best_init_probs = init_probs.copy()
            else:
                # Restore best parameters if the likelihood decreases
                trans_probs = best_trans_probs.copy()
                emission_probs = best_emission_probs.copy()
                init_probs = best_init_probs.copy()

            gamma_vals = np.zeros((seq_length, num_states))
            xi_vals = np.zeros((seq_length - 1, num_states, num_states))

            for t in range(seq_length):
                gamma_sum = np.sum(alpha_vals[t] * beta_vals[t]) + epsilon
                gamma_vals[t] = (alpha_vals[t] * beta_vals[t]) / gamma_sum

            for t in range(seq_length - 1):
                xi_sum = np.sum(np.outer(alpha_vals[t], beta_vals[t + 1]) * trans_probs * emission_probs[:, obs_seq[t + 1]]) + epsilon
                for i in range(num_states):
                    for j in range(num_states):
                        xi_vals[t, i, j] = (alpha_vals[t, i] * trans_probs[i, j] * emission_probs[j, obs_seq[t + 1]] * beta_vals[t + 1, j]) / xi_sum

            # Update initial probabilities
            init_probs = gamma_vals[0] + epsilon

            # Update transition probabilities
            for i in range(num_states):
                trans_probs[i] = np.sum(xi_vals[:, i, :], axis=0) / (np.sum(gamma_vals[:-1, i]) + epsilon)

            # Update emission probabilities
            for i in range(num_states):
                for k in range(num_obs):
                    mask = np.array(obs_seq) == k
                    emission_sum = np.sum(gamma_vals[mask, i])
                    state_sum = np.sum(gamma_vals[:, i]) + epsilon
                    emission_probs[i, k] = emission_sum / state_sum

            # Check for convergence based on likelihood improvement
            if np.abs(best_likelihood - current_likelihood) < tolerance:
                break

    return best_trans_probs, best_emission_probs, best_init_probs

# Observation sequence encoding
def encode_obs_seq(obs_seq, obs_labels):
    return np.array([obs_labels.index(obs) for obs in obs_seq])

# Get states and observation space
def get_states_and_obs(state_labels, obs_labels):
    return len(state_labels), len(obs_labels)

if __name__ == "__main__":
    # Define states and observations
    states, obs_labels = ['H', 'L'], ['R', 'D', 'E', 'F', 'G']
    num_states, num_obs = get_states_and_obs(states, obs_labels)

    # Observation sequences
    obs_sequences = [
        ['R', 'R', 'D', 'D', 'R', 'D', 'R'],
        ['D', 'R', 'D', 'D', 'R', 'R', 'D'],
        ['R', 'D', 'D', 'D', 'R', 'D', 'R'],
        ['R', 'R', 'R', 'D', 'D', 'R', 'D'],
        ['D', 'R', 'D', 'D', 'D', 'D', 'R']
    ]
    encoded_obs_sequences = [encode_obs_seq(seq, obs_labels) for seq in obs_sequences]
 # Initialize probabilities
    trans_probs = np.random.rand(num_states, num_states)
    trans_probs /= trans_probs.sum(axis=1, keepdims=True)

    emission_probs = np.random.rand(num_states, num_obs)
    emission_probs /= emission_probs.sum(axis=1, keepdims=True)
init_probs = steady_state_finder(trans_probs)

    # Run Baum-Welch with memory mechanism
    trans_probs, emission_probs, init_probs = baum_welch_algorithm_with_memory(
        encoded_obs_sequences, trans_probs, emission_probs, init_probs, num_states, num_obs
    )

    # Output the results
    print("Transition Probabilities:", trans_probs)
    print("Emission Probabilities:", emission_probs)
    print("Initial Probabilities:", init_probs)
, 'R']
    ]
    encoded_obs_sequences = [encode_obs_seq(seq, obs_labels) for seq in obs_sequences]

    trans_probs = np.random.rand(num_states, num_states)
    trans_probs /= trans_probs.sum(axis=1, keepdims=True)

    emission_probs = np.random.rand(num_states, num_obs)
    emission_probs /= emission_probs.sum(axis=1, keepdims=True)

    init_probs = steady_state_finder(trans_probs)

    trans_probs, emission_probs, init_probs = baum_welch_algorithm(
        encoded_obs_sequences, trans_probs, emission_probs, init_probs, num_states, num_obs
    )

    print("Transition Probabilities:", trans_probs)
    print("Emission Probabilities:", emission_probs)
    print("Initial Probabilities:", init_probs)
