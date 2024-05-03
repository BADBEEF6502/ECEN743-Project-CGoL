import numpy as np
import matplotlib.pyplot as plt

# Shows work for all of the sums and returns total memory consumed in bytes,
def compute_memory(size, replay_dim, action_replay_dim, action_dim, Q_net_multiplier):
    # Constants - hard coded.
    INT32_BYTES = 4

    # --- CGL ---
    world_gpu  = size
    stable_gpu = size
    result_gpu = size
    sim_tot = world_gpu + stable_gpu + result_gpu

    # --- DQN ---
    # Replay Buffer
    state      = replay_dim * size
    action     = replay_dim * action_replay_dim * INT32_BYTES
    next_state = replay_dim * size
    reward     = replay_dim * INT32_BYTES
    replay_buffer_tot = state + action + next_state + reward

    # Q Network
    l1 = size * action_dim * Q_net_multiplier
    l2 = (action_dim * Q_net_multiplier) ** 2
    l3 = action_dim * Q_net_multiplier * action_dim
    q_network_tot = l1 + l2 + l3

    # Q Target Network
    q_target_network_tot = q_network_tot

    # --- Total Memory ---
    mem_tot = q_target_network_tot + q_network_tot + replay_buffer_tot + sim_tot
    return mem_tot

# Compute the actions of the neural network, may become more complex later on.
def compute_action_dim(size):
    return size + 1

# Hyperparameters for tuning.
NET_MULTIPLIER    = 2       # User argument, hard coded in DQN.
MAX_SIDE          = 200     # Side is a user argument from command line.
action_replay_dim = 1       # User argument, hard coded in DQN.
replay_dim        = 1e5     # User argument, from command line.

# Produce and collect data.
x = []
y = []
for side in range(1, MAX_SIDE):
    size = side ** 2
    action_dim = compute_action_dim(size)
    mem_consumed = compute_memory(size, replay_dim, action_replay_dim, action_dim, NET_MULTIPLIER)
    x.append(side)
    y.append(mem_consumed)

# Create graph.
# Convert list to numpy arrays.
x = np.asarray(x)
y = np.asarray(y)
y /= 1073741824 # Convert bytes to GB (1024 ** 3).

# Fit a polynomial trendline (linear regression)
coefficients = np.polyfit(x, np.log(y), 1, w=np.sqrt(y))
trendline = np.exp(coefficients[0] * x) * np.exp(coefficients[1])

# Plot the data points
plt.scatter(x, y)

# Plot the trendline
plt.plot(x, trendline, color='red', label=f'Trendline: y = e^{coefficients[0]:.2f}x + e^{coefficients[1]:.2f}')

# Add axis titles
plt.title(f'GPU Memory Consumption Sweep\nNetwork Multiplier={NET_MULTIPLIER}, Experience Replay Buffer Size={replay_dim}')
plt.xlabel('Simulation Enviornment Side Length')
plt.ylabel('GPU Memory Consumed in GB')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()