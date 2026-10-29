import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
print("Ramith H 5T 1WN24CS222")
# Reproducible results
random.seed(10)
np.random.seed(10)

# 1. Create Wireless Sensor Network
G = nx.Graph()

edges = [
    (0, 1, 2), (0, 2, 3), (1, 3, 2),
    (2, 3, 1), (1, 4, 4), (3, 4, 2),
    (3, 5, 3), (4, 6, 2), (5, 6, 2)
]

for u, v, distance in edges:
    G.add_edge(u, v, distance=distance)

SOURCE = 0
SINK = 6
pos = nx.spring_layout(G, seed=10)

# 2. ACO Parameters
ants = 20
iterations = 50
alpha = 1.0
beta = 2.0
rho = 0.3

pheromone = {
    tuple(sorted((u, v))): 1.0
    for u, v in G.edges()
}

best_path = None
best_cost = float("inf")

convergence = []
iteration_best = []

# 3. Helper Functions
def edge_key(u, v):
    return tuple(sorted((u, v)))


def path_cost(path):
    return sum(
        G[u][v]["distance"]
        for u, v in zip(path[:-1], path[1:])
    )


# 4. Construct a Route Using Ants
def construct_path():
    current = SOURCE
    path = [current]
    visited = {current}

    while current != SINK:
        neighbors = [
            n for n in G.neighbors(current)
            if n not in visited
        ]

        if not neighbors:
            return None

        weights = []

        for n in neighbors:
            tau = pheromone[edge_key(current, n)]
            distance = G[current][n]["distance"]
            eta = 1.0 / distance

            weights.append(
                (tau ** alpha) * (eta ** beta)
            )

        probabilities = np.array(weights, dtype=float)
        probabilities /= probabilities.sum()

        next_node = random.choices(
            neighbors,
            weights=probabilities,
            k=1
        )[0]

        path.append(next_node)
        visited.add(next_node)
        current = next_node

    return path


# 5. Run Ant Colony Optimization
for iteration in range(iterations):
    solutions = []

    for _ in range(ants):
        path = construct_path()

        if path is not None:
            cost = path_cost(path)
            solutions.append((path, cost))

            if cost < best_cost:
                best_cost = cost
                best_path = path.copy()

    # Pheromone evaporation
    for edge in pheromone:
        pheromone[edge] *= (1 - rho)

    # Pheromone deposition
    for path, cost in solutions:
        deposit = 1.0 / cost

        for u, v in zip(path[:-1], path[1:]):
            pheromone[edge_key(u, v)] += deposit

    # Record convergence
    convergence.append(best_cost)

    if solutions:
        current_best = min(cost for path, cost in solutions)
        iteration_best.append(current_best)
    else:
        iteration_best.append(float("nan"))


# 6. Display Minimal Output
print("Best Route:", best_path)
print("Total Route Distance:", best_cost)

# 7. Draw Graphs
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# Network topology
nx.draw_networkx_edges(
    G, pos, ax=ax[0],
    edge_color="gray", width=1.5
)

nx.draw_networkx_nodes(
    G, pos, ax=ax[0],
    node_color="lightblue", node_size=800
)

nx.draw_networkx_labels(
    G, pos, ax=ax[0], font_size=12
)

nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=nx.get_edge_attributes(G, "distance"),
    ax=ax[0]
)

route_edges = list(
    zip(best_path[:-1], best_path[1:])
)

nx.draw_networkx_edges(
    G, pos, ax=ax[0],
    edgelist=route_edges,
    edge_color="red", width=3
)

ax[0].set_title("ACO Wireless Sensor Routing")
ax[0].axis("off")

# Convergence graph
ax[1].plot(
    range(1, iterations + 1),
    convergence,
    label="Best so far",
    color="blue",
    linewidth=2
)

ax[1].plot(
    range(1, iterations + 1),
    iteration_best,
    label="Best in each iteration",
    color="orange",
    linestyle="--",
    linewidth=1.5
)

ax[1].set_title("ACO Convergence")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Route Distance")
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()