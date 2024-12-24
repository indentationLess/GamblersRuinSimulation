import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def createMarkovChainDiagram(N=5, p=0.45):
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

    plt.title(f"markov chain for gambler's Ruin (p={p})")
    plt.axis("off")


def simulatePath(N=5, p=0.45, initialState=2):
    currentState = initialState
    path = [currentState]

    while 0 < currentState < N:
        if np.random.random() < p:
            currentState += 1
        else:
            currentState -= 1
        path.append(currentState)

    return path


plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
createMarkovChainDiagram(N=5, p=0.45)

plt.subplot(2, 1, 2)
initialState = 2
N = 5
numPaths = 5

for i in range(numPaths):
    path = simulatePath(N=N, p=0.45, initialState=initialState)
    steps = range(len(path))
    finalState = path[-1]
    color = "green" if finalState == N else "red"
    plt.plot(
        steps,
        path,
        marker="o",
        color=color,
        alpha=0.6,
        label=f'Path {i+1} - {"Won" if finalState == N else "lost"}',
    )

plt.axhline(y=N, color="g", linestyle="--", alpha=0.3, label="goal")
plt.axhline(y=0, color="r", linestyle="--", alpha=0.3, label="ruin")
plt.axhline(
    y=initialState, color="b", linestyle="--", alpha=0.3, label="starting Point"
)

plt.grid(True, alpha=0.3)
plt.title(f"sample Paths (Starting from state {initialState})")
plt.xlabel("steps")
plt.ylabel("state")
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
print(f"\ntheoretical probabilities starting from state {i}:")
print(f"probability of ruin: {probRuin(i, N, 0.45):.4f}")
print(f"probability of reaching goal: {1 - probRuin(i, N, 0.45):.4f}")
