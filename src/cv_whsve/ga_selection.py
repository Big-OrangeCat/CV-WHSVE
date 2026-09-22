"""Deterministic genetic-algorithm selection from OOF probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


@dataclass
class GAResult:
    selected_names: list[str]
    best_fitness: float
    generations_run: int
    history: list[float]


def _ensure_nonempty(chromosome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if not chromosome.any():
        chromosome[rng.integers(0, chromosome.size)] = 1
    return chromosome


def _fitness(chromosome: np.ndarray, probabilities: np.ndarray, y: np.ndarray) -> float:
    selected = np.flatnonzero(chromosome)
    score = probabilities[:, selected].mean(axis=1)
    pred = (score >= 0.5).astype(int)
    return float(0.6 * f1_score(y, pred, zero_division=0) + 0.4 * roc_auc_score(y, score))


def select_classifier_subset(
    probability_by_model: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    random_state: int = 42,
    population_size: int = 100,
    max_generations: int = 100,
    crossover_probability: float = 0.8,
    mutation_probability: float = 0.3,
    early_stop_generations: int = 5,
    minimum_relative_improvement: float = 0.001,
) -> GAResult:
    """Select a classifier subset using 0.6*F1 + 0.4*AUROC fitness."""
    names = list(probability_by_model)
    probabilities = np.column_stack([probability_by_model[name] for name in names])
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(random_state)
    population = rng.integers(0, 2, size=(population_size, len(names)), dtype=np.int8)
    population = np.asarray([_ensure_nonempty(row, rng) for row in population])

    best = None
    best_fitness = -np.inf
    no_improvement = 0
    history: list[float] = []

    for generation in range(max_generations):
        fitnesses = np.asarray([_fitness(row, probabilities, y) for row in population])
        order = np.argsort(fitnesses)[::-1]
        generation_best = float(fitnesses[order[0]])
        history.append(generation_best)
        relative_gain = (
            (generation_best - best_fitness) / max(abs(best_fitness), 1e-12)
            if np.isfinite(best_fitness)
            else np.inf
        )
        if generation_best > best_fitness:
            best_fitness = generation_best
            best = population[order[0]].copy()
        if relative_gain >= minimum_relative_improvement:
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= early_stop_generations:
            break

        progress = generation / max(max_generations - 1, 1)
        if progress <= 0.2:
            factor = 0.9 if relative_gain > 0 else 1.1
            crossover_probability = float(np.clip(crossover_probability * factor, 0.1, 0.95))
            mutation_probability = float(np.clip(mutation_probability * factor, 0.1, 0.5))
        else:
            tail_progress = (progress - 0.2) / 0.8
            crossover_probability = max(0.1, crossover_probability * (1.0 - tail_progress))
            mutation_probability = max(0.1, mutation_probability * (1.0 - tail_progress))

        elites = population[order[:2]].copy()
        next_population = [row.copy() for row in elites]
        while len(next_population) < population_size:
            contestants = rng.integers(0, population_size, size=3)
            parent1 = population[contestants[np.argmax(fitnesses[contestants])]].copy()
            contestants = rng.integers(0, population_size, size=3)
            parent2 = population[contestants[np.argmax(fitnesses[contestants])]].copy()
            if len(names) > 1 and rng.random() < crossover_probability:
                point = int(rng.integers(1, len(names)))
                child1 = np.concatenate([parent1[:point], parent2[point:]])
                child2 = np.concatenate([parent2[:point], parent1[point:]])
            else:
                child1, child2 = parent1, parent2
            for child in (child1, child2):
                flips = rng.random(len(names)) < mutation_probability
                child[flips] = 1 - child[flips]
                next_population.append(_ensure_nonempty(child, rng))
                if len(next_population) >= population_size:
                    break
        population = np.asarray(next_population[:population_size], dtype=np.int8)

    if best is None:
        raise RuntimeError("GA did not produce a valid classifier subset")
    selected_names = [name for name, keep in zip(names, best) if keep]
    return GAResult(selected_names, float(best_fitness), len(history), history)
