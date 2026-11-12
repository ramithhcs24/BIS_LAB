
import numpy as np
import matplotlib.pyplot as plt
import time
print("Ramith H 1WN24CS222")

# ============================================================
# 1. BENCHMARK FUNCTIONS
# ============================================================

def sphere(x):
    return np.sum(x ** 2)


def rastrigin(x):
    n = len(x)
    return 10 * n + np.sum(
        x ** 2 - 10 * np.cos(2 * np.pi * x)
    )


def ackley(x):
    n = len(x)
    return (
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
        + 20 + np.e
    )


# ============================================================
# 2. BOUNDARY CONTROL
# ============================================================

def boundary_control(x, lb, ub, rng):

    x = x.copy()
    width = ub - lb

    lower_mask = x < lb
    upper_mask = x > ub

    x[lower_mask] = (
        lb + width / 4 * rng.random(np.sum(lower_mask))
    )

    x[upper_mask] = (
        ub - width / 4 * rng.random(np.sum(upper_mask))
    )

    return x


# ============================================================
# 3. ROUNDING RULE
# ============================================================

def rounding_rule(si, fi, best_fitness, rng):

    evol = si.copy()

    total = fi + best_fitness

    pond = (
        0.5 if abs(total) < 1e-15
        else np.clip(1 - fi / total, 0, 1)
    )

    nr = rng.integers(1, 7)

    mask = rng.random(len(si)) <= pond

    evol[mask] = np.round(evol[mask], nr)

    return evol


# ============================================================
# 4. EXPLOITATION RULE
# ============================================================

def exploitation(si, prop, lb, ub, rng):

    shift_type = rng.integers(-1, 2)

    if shift_type == 0:
        comp_type = 1
    else:
        comp_type = rng.integers(0, 2)

    shift = np.roll(si, shift_type)

    if comp_type == 1:

        aux = shift - lb
        shift = ub - aux

    r = rng.random() * prop

    distance = (si - shift) * r

    evol = si - distance

    return evol


# ============================================================
# 5. EXPLORATION RULE
# ============================================================

def exploration(si, fi, sj, fj, prop, lb, ub, rng):

    shift_type = rng.integers(-1, 2)
    comp_type = rng.integers(0, 2)

    shift = np.roll(sj, shift_type)

    if comp_type == 1:

        aux = shift - lb
        shift = ub - aux

    total = fi + fj

    pond = (
        0.5 if abs(total) < 1e-15
        else np.clip(1 - fj / total, 0, 1)
    )

    r = (rng.random() * prop) - (prop / 2)

    evol = si.copy()

    mask = rng.random(len(si)) <= pond

    evol[mask] += r * shift[mask]

    return evol


# ============================================================
# 6. RECAA MAIN ALGORITHM
# ============================================================

def recaa(
    objective,
    dimensions=30,
    lb=-5.12,
    ub=5.12,
    ns=12,
    nne=6,
    nit=500,
    nel=2,
    prop=1.5,
    prob_gamma=0.05,
    seed=42
):

    rng = np.random.default_rng(seed)

    start_time = time.perf_counter()

    # Initialize smart-cells
    population = rng.uniform(
        lb, ub, size=(ns, dimensions)
    )

    fitness = np.array([
        objective(x) for x in population
    ])

    initial_best = np.min(fitness)

    convergence = [initial_best]

    diversity = [
        np.mean(np.std(population, axis=0))
    ]

    # ========================================================
    # OPTIMIZATION CYCLE
    # ========================================================

    for iteration in range(1, nit):

        # Preserve best elite solutions
        order = np.argsort(fitness)

        new_population = population[order[:nel]].copy()

        new_fitness = fitness[order[:nel]].copy()

        best_fitness = new_fitness[0]

        # Update remaining smart-cells
        for j in range(nel, ns):

            si = population[j].copy()
            fi = fitness[j]

            # Select random neighboring smart-cell
            random_index = rng.integers(0, ns)

            sj = population[random_index]
            fj = fitness[random_index]

            best_neighbor = None
            best_neighbor_fitness = np.inf

            # Generate neighborhood
            for k in range(nne):

                p = rng.random()

                # Rule selection from original RECAA
                if p <= 1 / 12:

                    evol = rounding_rule(
                        si, fi, best_fitness, rng
                    )

                elif p < 0.2:

                    evol = exploitation(
                        si, prop, lb, ub, rng
                    )

                else:

                    evol = exploration(
                        si, fi, sj, fj,
                        prop, lb, ub, rng
                    )

                # Apply boundary correction
                evol = boundary_control(
                    evol, lb, ub, rng
                )

                # Evaluate candidate
                evol_fitness = objective(evol)

                # Keep best neighbor
                if evol_fitness < best_neighbor_fitness:

                    best_neighbor = evol.copy()
                    best_neighbor_fitness = evol_fitness

            # Accept improved solution or random replacement
            if (
                rng.random() < prob_gamma
                or best_neighbor_fitness < fi
            ):

                new_population = np.vstack(
                    (new_population, best_neighbor)
                )

                new_fitness = np.append(
                    new_fitness, best_neighbor_fitness
                )

            else:

                new_population = np.vstack(
                    (new_population, si)
                )

                new_fitness = np.append(
                    new_fitness, fi
                )

        # Replace old population
        population = new_population
        fitness = new_fitness

        # Store best fitness
        current_best = np.min(fitness)

        convergence.append(current_best)

        # Store population diversity
        current_diversity = np.mean(
            np.std(population, axis=0)
        )

        diversity.append(current_diversity)

        # Progress display
        if iteration % 50 == 0:

            print(
                f"Iteration {iteration:3d}/{nit - 1} | "
                f"Best Fitness: {current_best:.8f} | "
                f"Diversity: {current_diversity:.6f}"
            )

    execution_time = time.perf_counter() - start_time

    # Final best solution
    best_index = np.argmin(fitness)

    best_solution = population[best_index].copy()

    best_fitness = fitness[best_index]

    return {
        "best_solution": best_solution,
        "best_fitness": best_fitness,
        "initial_best": initial_best,
        "convergence": convergence,
        "diversity": diversity,
        "execution_time": execution_time
    }


# ============================================================
# 7. EXPERIMENT
# ============================================================

if __name__ == "__main__":

    DIMENSIONS = 30
    ITERATIONS = 500

    # Select benchmark function
    objective = rastrigin

    print("\n========================================")
    print("       RECAA OPTIMIZATION RESULTS")
    print("========================================")

    print("Function       : Rastrigin")
    print("Dimensions     :", DIMENSIONS)
    print("Smart Cells    : 12")
    print("Neighbors      : 6")
    print("Iterations     :", ITERATIONS)

    print("----------------------------------------")

    result = recaa(
        objective=objective,
        dimensions=DIMENSIONS,
        lb=-5.12,
        ub=5.12,
        ns=12,
        nne=6,
        nit=ITERATIONS,
        nel=2,
        prop=1.5,
        prob_gamma=0.05,
        seed=42
    )

    # ========================================================
    # 8. FINAL RESULTS
    # ========================================================

    initial = result["initial_best"]
    final = result["best_fitness"]

    improvement = (
        (initial - final) / max(abs(initial), 1e-15)
    ) * 100

    print("\n========== FINAL RESULTS ==========")

    print(f"Initial Fitness : {initial:.8f}")
    print(f"Final Fitness   : {final:.8f}")
    print(f"Improvement     : {improvement:.2f}%")

    print(
        f"Execution Time  : "
        f"{result['execution_time']:.4f} seconds"
    )

    print("\nBest Solution (first 10 variables):")

    print(np.round(result["best_solution"][:10], 6))

    print("===================================\n")

    # ========================================================
    # 9. GRAPH 1 - CONVERGENCE
    # ========================================================

    plt.figure(figsize=(9, 5))

    plt.plot(
        result["convergence"],
        color="blue",
        linewidth=2
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--",
        label="Theoretical Global Minimum"
    )

    plt.xlabel("Iterations")
    plt.ylabel("Best Fitness Value")

    plt.title("RECAA - Convergence Curve")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "RECAA_Convergence.png",
        dpi=300
    )

    plt.show()

    # ========================================================
    # 10. GRAPH 2 - POPULATION DIVERSITY
    # ========================================================

    plt.figure(figsize=(9, 5))

    plt.plot(
        result["diversity"],
        color="green",
        linewidth=2
    )

    plt.xlabel("Iterations")
    plt.ylabel("Average Population Diversity")

    plt.title("RECAA - Population Diversity")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "RECAA_Diversity.png",
        dpi=300
    )

    plt.show()

    print("Both graphs saved successfully!")
