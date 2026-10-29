
import numpy as np
import matplotlib.pyplot as plt
print("Ramith H  5T  1WN24CS222")
print("Cuckoo Search Algorithm")

# 1. Generator data
N = 3
demand = 850

Pmin = np.array([100, 50, 50], dtype=float)
Pmax = np.array([500, 200, 300], dtype=float)

a = np.array([100, 120, 150])
b = np.array([5.0, 4.8, 4.5])
c = np.array([0.004, 0.006, 0.007])
d = np.array([100, 120, 150])
e = np.array([0.05, 0.045, 0.04])


# 2. Make generation satisfy demand and limits
def repair(P):
    P = np.clip(P, Pmin, Pmax)

    for _ in range(100):
        error = demand - P.sum()

        if abs(error) < 1e-9:
            break

        if error > 0:
            room = Pmax - P
        else:
            room = P - Pmin

        if room.sum() < abs(error):
            break

        P += np.sign(error) * np.minimum(
            room, abs(error) * room / room.sum()
        )

    return np.clip(P, Pmin, Pmax)


# 3. Fuel cost
def cost(P):
    return np.sum(
        a + b * P + c * P**2
        + d * np.abs(np.sin(e * (Pmin - P)))
    )


# 4. Cuckoo Search
def cuckoo_search(seed, modified=False, nests=25, iterations=250):
    rng = np.random.default_rng(seed)

    population = np.array([
        repair(rng.uniform(Pmin, Pmax))
        for _ in range(nests)
    ])

    scores = np.array([cost(p) for p in population])
    best_idx = np.argmin(scores)
    best = population[best_idx].copy()
    best_cost = scores[best_idx]
    history = []

    for t in range(iterations):
        progress = t / iterations

        # Modified algorithm gradually reduces step size
        if modified:
            step_size = 0.12 * (1 - progress) + 0.002
        else:
            step_size = 0.04

        for i in range(nests):
            step = rng.normal(0, 1, N)

            candidate = repair(
                population[i]
                + step_size * step * (Pmax - Pmin)
            )
            candidate_cost = cost(candidate)

            if candidate_cost < scores[i]:
                population[i] = candidate
                scores[i] = candidate_cost

        # Modified version adds local search
        if modified:
            for _ in range(5):
                candidate = repair(
                    best + rng.normal(0, 1, N)
                    * 0.02 * (1 - progress) * (Pmax - Pmin)
                )

                candidate_cost = cost(candidate)

                if candidate_cost < best_cost:
                    best = candidate.copy()
                    best_cost = candidate_cost

        # Replace some poor nests
        worst = np.argsort(scores)[-5:]

        for i in worst:
            if rng.random() < 0.5:
                population[i] = repair(
                    rng.uniform(Pmin, Pmax)
                )
                scores[i] = cost(population[i])

        idx = np.argmin(scores)

        if scores[idx] < best_cost:
            best = population[idx].copy()
            best_cost = scores[idx]

        history.append(best_cost)

    return best, best_cost, history


# 5. Run both algorithms with the same initial seed
csa_P, csa_cost, csa_history = cuckoo_search(
    seed=42, modified=False
)

mcsa_P, mcsa_cost, mcsa_history = cuckoo_search(
    seed=42, modified=True
)


# 6. Print understandable results
print("=" * 48)
print(" ECONOMIC POWER DISPATCH RESULTS")
print("=" * 48)
print(f"Required demand: {demand} MW\n")

print("Generator outputs:")
print(f"{'Generator':<12}{'CSA (MW)':>12}{'MCSA (MW)':>12}")

for i in range(N):
    print(
        f"{i+1:<12}{csa_P[i]:>12.2f}{mcsa_P[i]:>12.2f}"
    )

print(f"\nCSA total generation : {csa_P.sum():.4f} MW")
print(f"MCSA total generation: {mcsa_P.sum():.4f} MW")

print(f"\nCSA fuel cost : {csa_cost:.4f} units/hour")
print(f"MCSA fuel cost: {mcsa_cost:.4f} units/hour")

difference = csa_cost - mcsa_cost
improvement = difference / csa_cost * 100

print(f"\nCost difference: {difference:.4f} units/hour")
print(f"Cost reduction: {improvement:.4f}%")

if difference > 0:
    print("Result: MCSA found a lower-cost solution in this run.")
elif difference < 0:
    print("Result: CSA found a lower-cost solution in this run.")
else:
    print("Result: Both algorithms found the same cost.")

print("=" * 48)


# 7. Two simple comparison graphs
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

# Convergence
ax[0].plot(csa_history, label="Standard CSA")
ax[0].plot(mcsa_history, label="Modified CSA")
ax[0].set_title("Fuel Cost vs Iterations")
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Best fuel cost")
ax[0].legend()
ax[0].grid(alpha=0.3)

# Generator outputs
x = np.arange(N)
width = 0.35

ax[1].bar(x - width/2, csa_P, width, label="CSA")
ax[1].bar(x + width/2, mcsa_P, width, label="MCSA")
ax[1].set_xticks(x)
ax[1].set_xticklabels(["Generator 1", "Generator 2", "Generator 3"])
ax[1].set_ylabel("Power (MW)")
ax[1].set_title("Power Generation Comparison")
ax[1].legend()
ax[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
