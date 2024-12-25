import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Configurable parameters
N = 5
initialP = 0.45  # Initial probability of winning
decay = 0.95  # Decay factor for the probability of winning
initialState = 2
numPaths = 5


def createMarkovChainDiagram(N, p):
    G = nx.DiGraph()
    nodes = range(N + 1)
    G.add_nodes_from(nodes)
    pos = {i: (i, 0) for i in nodes}

    for i in range(1, N):
        G.add_edge(i, i + 1)
        G.add_edge(i, i - 1)
    G.add_edge(0, 0)
    G.add_edge(N, N)

    nx.draw(
        G,
        pos,
        with_labels=False,
        node_color="lightblue",
        node_size=2000,
        edge_color="gray",
        connectionstyle="arc3,rad=0.2",
        arrowsize=20,
    )

    nx.draw_networkx_labels(G, pos, labels={i: str(i) for i in nodes})
    edgeLabels = {(i, i + 1): f"p={p}" for i in range(1, N)}
    edgeLabels.update({(i, i - 1): f"q={1 - p}" for i in range(1, N)})
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edgeLabels)

    plt.title(f"Markov Chain for Gambler's Ruin (p={p})")
    plt.axis("off")


def simulatePath(N, p, initialState, decay):
    currentState = initialState
    path = [currentState]
    currentP = p

    while 0 < currentState < N:
        if np.random.random() < currentP:
            currentState += 1
        else:
            currentState -= 1
        path.append(currentState)
        currentP *= decay

    return path


plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
createMarkovChainDiagram(N=N, p=initialP)

plt.subplot(2, 1, 2)

for i in range(numPaths):
    path = simulatePath(N=N, p=initialP, initialState=initialState, decay=decay)
    steps = range(len(path))
    finalState = path[-1]
    color = "green" if finalState == N else "red"
    plt.plot(
        steps,
        path,
        marker="o",
        color=color,
        alpha=0.6,
        label=f'Path {i+1} - {"Won" if finalState == N else "Lost"}',
    )

plt.axhline(y=N, color="g", linestyle="--", alpha=0.3, label="Goal")
plt.axhline(y=0, color="r", linestyle="--", alpha=0.3, label="Ruin")
plt.axhline(
    y=initialState, color="b", linestyle="--", alpha=0.3, label="Starting Point"
)

plt.grid(True, alpha=0.3)
plt.title(f"Sample Paths (Starting from state {initialState})")
plt.xlabel("Steps")
plt.ylabel("State")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.ylim(-0.5, N + 0.5)

plt.tight_layout()
plt.show()


def probRuin(i, N, p):
    q = 1 - p
    if p == 0.5:
        return 1 - i / N
    else:
        return (1 - (q / p) ** i) / (1 - (q / p) ** N)


i = initialState
print(f"\nTheoretical probabilities starting from state {i}:")
print(f"Probability of ruin: {1 - probRuin(i, N, initialP):.4f}")
print(f"Probability of reaching goal: {probRuin(i, N, initialP):.4f}")
