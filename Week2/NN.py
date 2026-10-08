import random
import math

# ---------------------------------------------------
# 1. TRAINING DATA - XOR PROBLEM
# ---------------------------------------------------

inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

targets = [0, 1, 1, 0]

# ---------------------------------------------------
# 2. PSO PARAMETERS
# ---------------------------------------------------

POPULATION_SIZE = 5
ITERATIONS = 100

W = 0.5
C1 = 1.2
C2 = 1.2

# ---------------------------------------------------
# 3. NEURAL NETWORK PARAMETERS
# ---------------------------------------------------

# 2 Input -> 2 Hidden -> 1 Output
#
# Input -> Hidden weights = 4
# Hidden biases           = 2
# Hidden -> Output weights = 2
# Output bias             = 1
#
# Total parameters = 9

NUM_WEIGHTS = 9

# ---------------------------------------------------
# 4. SIGMOID ACTIVATION FUNCTION
# ---------------------------------------------------

def sigmoid(x):

    if x < -500:
        return 0

    if x > 500:
        return 1

    return 1 / (1 + math.exp(-x))

# ---------------------------------------------------
# 5. NEURAL NETWORK
# ---------------------------------------------------

def neural_network(weights, input_data):

    hidden_weights = weights[0:4]
    hidden_biases = weights[4:6]
    output_weights = weights[6:8]
    output_bias = weights[8]

    hidden_outputs = []

    for h in range(2):

        total = hidden_biases[h]

        for i in range(2):

            weight_index = h * 2 + i

            total += (
                input_data[i]
                * hidden_weights[weight_index]
            )

        hidden_outputs.append(
            sigmoid(total)
        )

    output = output_bias

    for h in range(2):

        output += (
            hidden_outputs[h]
            * output_weights[h]
        )

    output = sigmoid(output)

    return output

# ---------------------------------------------------
# 6. FITNESS FUNCTION
# ---------------------------------------------------

def fitness(weights):

    total_error = 0

    for i in range(len(inputs)):

        prediction = neural_network(
            weights,
            inputs[i]
        )

        error = (
            targets[i] - prediction
        ) ** 2

        total_error += error

    return total_error / len(inputs)

# ---------------------------------------------------
# 7. INITIALIZE PARTICLES
# ---------------------------------------------------

def create_particles():

    particles = []

    for _ in range(POPULATION_SIZE):

        position = []

        for _ in range(NUM_WEIGHTS):

            position.append(
                random.uniform(-2, 2)
            )

        velocity = []

        for _ in range(NUM_WEIGHTS):

            velocity.append(
                random.uniform(-1, 1)
            )

        current_fitness = fitness(position)

        particle = {

            "position": position,

            "velocity": velocity,

            "pbest": position.copy(),

            "pbest_fitness": current_fitness
        }

        particles.append(particle)

    return particles

# ---------------------------------------------------
# 8. UPDATE VELOCITY
# ---------------------------------------------------

def update_velocity(particle, global_best):

    for i in range(NUM_WEIGHTS):

        r1 = random.random()
        r2 = random.random()

        inertia = (
            W
            * particle["velocity"][i]
        )

        cognitive = (
            C1
            * r1
            * (
                particle["pbest"][i]
                - particle["position"][i]
            )
        )

        social = (
            C2
            * r2
            * (
                global_best[i]
                - particle["position"][i]
            )
        )

        particle["velocity"][i] = (
            inertia
            + cognitive
            + social
        )

# ---------------------------------------------------
# 9. UPDATE POSITION
# ---------------------------------------------------

def update_position(particle):

    for i in range(NUM_WEIGHTS):

        particle["position"][i] += (
            particle["velocity"][i]
        )

# ---------------------------------------------------
# 10. PARTICLE SWARM OPTIMIZATION
# ---------------------------------------------------

def particle_swarm_optimization():

    particles = create_particles()

    global_best = None
    global_best_fitness = float("inf")

    for particle in particles:

        if (
            particle["pbest_fitness"]
            < global_best_fitness
        ):

            global_best_fitness = (
                particle["pbest_fitness"]
            )

            global_best = (
                particle["pbest"].copy()
            )

    for iteration in range(ITERATIONS):

        for particle in particles:

            update_velocity(
                particle,
                global_best
            )

            update_position(
                particle
            )

            current_fitness = fitness(
                particle["position"]
            )

            if (
                current_fitness
                < particle["pbest_fitness"]
            ):

                particle["pbest"] = (
                    particle["position"].copy()
                )

                particle["pbest_fitness"] = (
                    current_fitness
                )

            if (
                current_fitness
                < global_best_fitness
            ):

                global_best_fitness = (
                    current_fitness
                )

                global_best = (
                    particle["position"].copy()
                )

        if iteration % 10 == 0:

            print(
                "Iteration:",
                iteration,
                "Best Error:",
                round(
                    global_best_fitness,
                    6
                )
            )

    return global_best, global_best_fitness

# ---------------------------------------------------
# 11. RUN PSO
# ---------------------------------------------------

best_weights, best_error = (
    particle_swarm_optimization()
)

# ---------------------------------------------------
# 12. DISPLAY FINAL RESULT
# ---------------------------------------------------

print("\n-----------------------------")
print("FINAL RESULT")
print("-----------------------------")

print(
    "Best Error:",
    round(best_error, 6)
)

print("\nNeural Network Predictions:")

for i in range(len(inputs)):

    prediction = neural_network(
        best_weights,
        inputs[i]
    )

    print(
        "Input:",
        inputs[i],
        "Target:",
        targets[i],
        "Prediction:",
        round(prediction, 4)
    )