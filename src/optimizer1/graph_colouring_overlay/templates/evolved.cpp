/**
 * @file evolved.cpp
 * @brief Seed implementation of the evolved algorithm slot.
 *
 * The proposer is expected to mutate this file. Helpful patterns:
 *   - Warm-start with DSatur or Welsh-Powell (`colour_with_dsatur`,
 *     `colour_with_welsh_powell`) and run a tabu / SA inner loop on the
 *     resulting palette.
 *   - Density-aware dispatch: choose a different inner search depending
 *     on graph density (sparse → DSatur is often near-optimal; dense →
 *     metaheuristic search pays off).
 *   - Path-relinking, HEA, or DGE on top of TabuCol (declared in
 *     `tabu.h`).
 *
 * Eval contract:
 *   - colors_used (PRIMARY, lower is better, must be a valid colouring of `graph`)
 *   - runtime_ms (TIEBREAKER, lower is better — only matters when
 *     colors_used ties between candidates)
 *
 * Do NOT call algorithms outside the graph_colouring namespace, read the
 * known-optimal metadata at runtime, or hardcode a known-optimal table.
 */

#include "evolved.h"

#include "tabu.h"

namespace graph_colouring {

std::vector<int> colour_with_evolved(const Graph &graph) {
    return colour_with_tabu(graph);
}

}  // namespace graph_colouring
