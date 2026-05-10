/**
 * @file evolved.h
 * @brief Evolved graph-colouring algorithm slot — proposer-mutated entry point.
 *
 * This file is injected by the Optimizer1 graph-colouring overlay (see
 * optimizer1.graph_colouring_overlay). The eval harness invokes
 * `build/benchmark_runner --algorithm evolved` and dispatches to
 * colour_with_evolved(). Every iteration the proposer is free to rewrite
 * evolved.cpp's body — hybridise existing algorithms (DSatur warm-start +
 * Tabu inner loop, Welsh-Powell post-processing, density-aware switching),
 * add path-relinking / HEA, etc. Do not change this signature; the harness
 * keys off it.
 */

#pragma once

#include <vector>

#include "../utils.h"

namespace graph_colouring {

/**
 * @brief Evolved heuristic entry point for the graph-colouring benchmark.
 *
 * The seed implementation delegates to colour_with_tabu so the iter-0
 * Pareto point is the upstream TabuCol baseline. The proposer is expected
 * to replace the body with non-trivial algorithmic mutations.
 *
 * @param graph The input graph to colour.
 * @return std::vector<int> Colour assignments where result[v] is the
 *         colour of vertex v (0-indexed).
 */
std::vector<int> colour_with_evolved(const Graph &graph);

}  // namespace graph_colouring
