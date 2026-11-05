
import numpy as np
import matplotlib.pyplot as plt
import os
print("Ramith H 5T 1WN24CS222")

# ============================================================
# GREY WOLF OPTIMIZATION FOR RENEWABLE ENERGY ALLOCATION
# IEEE 33-BUS DISTRIBUTION SYSTEM
# ============================================================

np.random.seed(42)
os.makedirs("GWO_Results", exist_ok=True)

# ============================================================
# 1. SYSTEM DATA
# ============================================================

BASE_MVA = 10.0
BASE_KV = 12.66
Z_BASE = BASE_KV ** 2 / BASE_MVA
N_BUS = 33

# Branch data: From, To, Resistance, Reactance
branches = np.array([
    [1, 2, .0922, .0470],
    [2, 3, .4930, .2511],
    [3, 4, .3660, .1864],
    [4, 5, .3811, .1941],
    [5, 6, .8190, .7070],
    [6, 7, .1872, .6188],
    [7, 8, 1.7114, 1.2351],
    [8, 9, 1.0300, .7400],
    [9, 10, 1.0440, .7400],
    [10, 11, .1966, .0650],
    [11, 12, .3744, .1238],
    [12, 13, 1.4680, 1.1550],
    [13, 14, .5416, .7129],
    [14, 15, .5910, .5260],
    [15, 16, .7463, .5450],
    [16, 17, 1.2890, 1.7210],
    [17, 18, .7320, .5740],
    [2, 19, .1640, .1565],
    [19, 20, 1.5042, 1.3554],
    [20, 21, .4095, .4784],
    [21, 22, .7089, .9373],
    [3, 23, .4512, .3083],
    [23, 24, .8980, .7091],
    [24, 25, .8960, .7011],
    [6, 26, .2030, .1034],
    [26, 27, .2842, .1447],
    [27, 28, 1.0590, .9337],
    [28, 29, .8042, .7006],
    [29, 30, .5075, .2585],
    [30, 31, .9744, .9630],
    [31, 32, .3105, .3619],
    [32, 33, .3410, .5302]
])

# Bus loads: kW and kVAR
loads = np.array([
    [0, 0],
    [100, 60], [90, 40], [120, 80],
    [60, 30], [60, 20], [200, 100],
    [200, 100], [60, 20], [60, 20],
    [45, 30], [60, 35], [60, 35],
    [120, 80], [60, 10], [60, 20],
    [60, 20], [90, 40], [90, 40],
    [90, 40], [90, 40], [90, 40],
    [90, 50], [420, 200], [420, 200],
    [60, 25], [60, 25], [60, 20],
    [120, 70], [200, 600], [150, 70],
    [210, 100], [60, 40]
])

FROM = branches[:, 0].astype(int) - 1
TO = branches[:, 1].astype(int) - 1

R = branches[:, 2] / Z_BASE
X = branches[:, 3] / Z_BASE

P_LOAD = loads[:, 0] / (BASE_MVA * 1000)
Q_LOAD = loads[:, 1] / (BASE_MVA * 1000)


# ============================================================
# 2. DAILY LOAD AND RENEWABLE PROFILES
# ============================================================

hours = np.arange(24)

daily_load = np.array([
    .65, .60, .58, .56, .58, .65,
    .75, .85, .90, .88, .85, .82,
    .80, .82, .85, .90, .95, 1.00,
    .98, .95, .90, .82, .75, .68
])

seasons = ["Summer", "Monsoon", "Winter"]
season_load = [1.00, .95, .90]

solar_profiles = np.array([
    np.maximum(0, np.sin(np.pi * (hours - 6) / 12)),
    np.maximum(0, np.sin(np.pi * (hours - 6) / 12)) * .85,
    np.maximum(0, np.sin(np.pi * (hours - 6) / 12)) * .70
])

wind_profiles = np.array([
    [.55, .58, .60, .62, .65, .60,
     .55, .50, .48, .45, .42, .40,
     .43, .46, .50, .55, .60, .65,
     .70, .68, .65, .62, .60, .58],

    [.70, .72, .75, .78, .75, .70,
     .65, .60, .55, .50, .48, .45,
     .50, .55, .60, .65, .70, .75,
     .78, .80, .78, .75, .72, .70],

    [.40, .42, .45, .48, .50, .52,
     .55, .58, .60, .62, .60, .58,
     .55, .52, .50, .48, .45, .42,
     .40, .42, .45, .48, .45, .42]
])


# ============================================================
# 3. LOAD FLOW USING FORWARD-BACKWARD SWEEP
# ============================================================

def load_flow(pv_bus, wt_bus, pv_size, wt_size,
              season, hour):

    p = P_LOAD * daily_load[hour] * season_load[season]
    q = Q_LOAD * daily_load[hour] * season_load[season]

    p = p.copy()
    q = q.copy()

    p[pv_bus - 1] -= (
        pv_size * solar_profiles[season, hour] / BASE_MVA
    )

    p[wt_bus - 1] -= (
        wt_size * wind_profiles[season, hour] / BASE_MVA
    )

    S = p + 1j * q
    voltage = np.ones(N_BUS, dtype=complex)

    for _ in range(100):

        current_bus = np.conj(S / voltage)
        branch_current = np.zeros(len(branches), dtype=complex)

        # Backward sweep
        for k in range(len(branches) - 1, -1, -1):

            child = TO[k]
            parent = FROM[k]

            branch_current[k] = current_bus[child]
            current_bus[parent] += branch_current[k]

        # Forward sweep
        new_voltage = np.ones(N_BUS, dtype=complex)

        for k in range(len(branches)):

            parent = FROM[k]
            child = TO[k]

            new_voltage[child] = (
                new_voltage[parent]
                - (R[k] + 1j * X[k]) * branch_current[k]
            )

        error = np.max(np.abs(new_voltage - voltage))
        voltage = new_voltage

        if error < 1e-9:
            break

    if not np.all(np.isfinite(voltage)):
        return None, None

    voltage_magnitude = np.abs(voltage)

    active_loss = (
        np.sum(np.abs(branch_current) ** 2 * R)
        * BASE_MVA
    )

    return voltage_magnitude, active_loss


# ============================================================
# 4. FITNESS FUNCTION
# ============================================================

def fitness(wolf):

    pv_bus = int(np.clip(round(wolf[0]), 2, 33))
    wt_bus = int(np.clip(round(wolf[1]), 2, 33))

    pv_size = np.clip(wolf[2], .05, 2.0)
    wt_size = np.clip(wolf[3], .05, 2.0)

    if pv_bus == wt_bus:
        return 1000.0

    total_deviation = 0
    total_penalty = 0
    count = 0

    for season in range(3):

        for hour in range(24):

            v, loss = load_flow(
                pv_bus, wt_bus,
                pv_size, wt_size,
                season, hour
            )

            if v is None:
                return 1000.0

            deviation = np.sum(np.abs(v[1:] - 1.0))

            violation = (
                np.sum(np.maximum(0, .95 - v[1:]) ** 2)
                + np.sum(np.maximum(0, v[1:] - 1.05) ** 2)
            )

            total_deviation += deviation
            total_penalty += violation
            count += 1

    return (
        total_deviation / count
        + 20 * total_penalty / count
    )


# ============================================================
# 5. GREY WOLF OPTIMIZATION ALGORITHM
# ============================================================

def GWO(population=25, iterations=20):

    dimension = 4

    lower = np.array([2, 2, .05, .05])
    upper = np.array([33, 33, 2.0, 2.0])

    wolves = np.random.uniform(
        lower, upper, (population, dimension)
    )

    alpha = np.zeros(dimension)
    beta = np.zeros(dimension)
    delta = np.zeros(dimension)

    alpha_score = np.inf
    beta_score = np.inf
    delta_score = np.inf

    convergence = []

    for iteration in range(iterations):

        for i in range(population):

            score = fitness(wolves[i])

            if score < alpha_score:

                delta_score, delta = beta_score, beta.copy()
                beta_score, beta = alpha_score, alpha.copy()

                alpha_score = score
                alpha = wolves[i].copy()

            elif score < beta_score:

                delta_score, delta = beta_score, beta.copy()

                beta_score = score
                beta = wolves[i].copy()

            elif score < delta_score:

                delta_score = score
                delta = wolves[i].copy()

        a = 2 - 2 * iteration / iterations

        # Position update using alpha, beta and delta
        for i in range(population):

            X = []

            for leader in [alpha, beta, delta]:

                r1 = np.random.rand(dimension)
                r2 = np.random.rand(dimension)

                A = 2 * a * r1 - a
                C = 2 * r2

                D = np.abs(C * leader - wolves[i])

                X.append(leader - A * D)

            wolves[i] = np.mean(X, axis=0)
            wolves[i] = np.clip(wolves[i], lower, upper)

        convergence.append(alpha_score)

        print(
            f"Iteration {iteration + 1:02d}/{iterations}"
            f" | Best Fitness: {alpha_score:.6f}"
        )

    return alpha, convergence


# ============================================================
# 6. RUN OPTIMIZATION
# ============================================================

print("\n" + "=" * 60)
print("       GREY WOLF OPTIMIZATION")
print("       RENEWABLE ENERGY ALLOCATION")
print("       IEEE 33-BUS DISTRIBUTION SYSTEM")
print("=" * 60)

best_solution, convergence = GWO(
    population=25,
    iterations=20
)

pv_bus = int(round(best_solution[0]))
wt_bus = int(round(best_solution[1]))

pv_size = best_solution[2]
wt_size = best_solution[3]


# ============================================================
# 7. PERFORMANCE EVALUATION
# ============================================================

base_voltages = []
optimized_voltages = []

base_losses = []
optimized_losses = []

for season in range(3):

    for hour in range(24):

        v_base, loss_base = load_flow(
            2, 3, 0, 0, season, hour
        )

        v_opt, loss_opt = load_flow(
            pv_bus, wt_bus,
            pv_size, wt_size,
            season, hour
        )

        base_voltages.append(v_base)
        optimized_voltages.append(v_opt)

        base_losses.append(loss_base)
        optimized_losses.append(loss_opt)

base_voltages = np.array(base_voltages)
optimized_voltages = np.array(optimized_voltages)

base_losses = np.array(base_losses)
optimized_losses = np.array(optimized_losses)

base_deviation = np.mean(
    np.abs(base_voltages[:, 1:] - 1)
)

opt_deviation = np.mean(
    np.abs(optimized_voltages[:, 1:] - 1)
)

base_min_voltage = np.min(base_voltages)
opt_min_voltage = np.min(optimized_voltages)

base_avg_loss = np.mean(base_losses)
opt_avg_loss = np.mean(optimized_losses)

voltage_improvement = (
    (base_deviation - opt_deviation)
    / base_deviation * 100
)

loss_reduction = (
    (base_avg_loss - opt_avg_loss)
    / base_avg_loss * 100
)


# ============================================================
# 8. GRAPH 1 - GWO CONVERGENCE
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, len(convergence) + 1),
    convergence,
    color="blue",
    linewidth=2,
    marker="o",
    markersize=4
)

plt.title("GWO Convergence Curve")
plt.xlabel("Number of Iterations")
plt.ylabel("Best Fitness Value")
plt.grid(True, alpha=.3)

plt.tight_layout()

plt.savefig(
    "GWO_Results/01_GWO_Convergence.png",
    dpi=300
)

plt.show()


# ============================================================
# 9. GRAPH 2 - VOLTAGE PROFILE
# ============================================================

base_profile = np.mean(base_voltages, axis=0)
optimized_profile = np.mean(optimized_voltages, axis=0)

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, 34),
    base_profile,
    "r--",
    linewidth=2,
    label="Before Optimization"
)

plt.plot(
    range(1, 34),
    optimized_profile,
    "g-o",
    markersize=3,
    linewidth=2,
    label="After GWO Optimization"
)

plt.axhline(
    .95,
    color="black",
    linestyle=":",
    label="Lower Voltage Limit"
)

plt.axhline(
    1.05,
    color="gray",
    linestyle=":",
    label="Upper Voltage Limit"
)

plt.title("IEEE 33-Bus Voltage Profile")
plt.xlabel("Bus Number")
plt.ylabel("Voltage Magnitude (p.u.)")
plt.xticks(range(1, 34))
plt.grid(True, alpha=.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    "GWO_Results/02_Voltage_Profile.png",
    dpi=300
)

plt.show()


# ============================================================
# 10. FINAL RESULTS - FORMATTED OUTPUT
# ============================================================

print("\n\n")
print("=" * 65)
print("              FINAL PROJECT REPORT")
print("=" * 65)

print("\n1. PROJECT DETAILS")
print("-" * 65)

print("Application          : Renewable Energy Allocation")
print("Algorithm            : Grey Wolf Optimization")
print("Distribution System  : IEEE 33-Bus")
print("Renewable Sources    : Solar PV and Wind Turbine")
print("Seasons Considered   : Summer, Monsoon, Winter")
print("Number of Iterations : 20")

print("\n2. OPTIMAL GENERATOR ALLOCATION")
print("-" * 65)

print(f"{'Parameter':<32}{'Optimal Result':>20}")
print("-" * 65)

print(f"{'Solar PV Bus':<32}{pv_bus:>20}")
print(f"{'Solar PV Capacity':<32}{pv_size:>17.4f} MW")
print(f"{'Wind Turbine Bus':<32}{wt_bus:>20}")
print(f"{'Wind Turbine Capacity':<32}{wt_size:>17.4f} MW")
print(f"{'Best Fitness Value':<32}{min(convergence):>20.6f}")

print("\n3. PERFORMANCE COMPARISON")
print("-" * 65)

print(
    f"{'Metric':<32}"
    f"{'Before':>15}"
    f"{'After':>15}"
)

print("-" * 65)

print(
    f"{'Voltage Deviation (p.u.)':<32}"
    f"{base_deviation:>15.5f}"
    f"{opt_deviation:>15.5f}"
)

print(
    f"{'Minimum Voltage (p.u.)':<32}"
    f"{base_min_voltage:>15.5f}"
    f"{opt_min_voltage:>15.5f}"
)

print(
    f"{'Active Power Loss (MW)':<32}"
    f"{base_avg_loss:>15.5f}"
    f"{opt_avg_loss:>15.5f}"
)

print("\n4. IMPROVEMENT ANALYSIS")
print("-" * 65)

print(
    f"Voltage Deviation Improvement : "
    f"{voltage_improvement:.2f}%"
)

print(
    f"Active Power Loss Reduction   : "
    f"{loss_reduction:.2f}%"
)

print("\n5. GENERATED GRAPH FILES")
print("-" * 65)

print("01. GWO_Results/01_GWO_Convergence.png")
print("02. GWO_Results/02_Voltage_Profile.png")

print("\n" + "=" * 65)
print("          OPTIMIZATION COMPLETED")
print("=" * 65)
