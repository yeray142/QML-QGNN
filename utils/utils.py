import numpy as np


def graph_to_list(nodes: list, fully_connected_edges: list[tuple], edge_weights: dict,
                  available_nodes: list[int], node_to_qubit_map: dict) -> list[float]:
    """
    Convert a graph into a list.
    :param nodes: list of graph nodes.
    :param fully_connected_edges: list of tuples of edges.
    :param edge_weights: dictionary of edge weights.
    :param available_nodes: list of available nodes.
    :param node_to_qubit_map: dictionary to map nodes and qubits.
    :return: list of values from graph.
    """
    vals = []
    for node in nodes:
        vals.append(int(node_to_qubit_map[node] in available_nodes) * np.pi)

    for edge in fully_connected_edges:
        vals.append(np.arctan(edge_weights[edge]))

    return vals


def compute_tour_length(nodes: list[tuple], tour: list[int]) -> float:
    """
    Compute length of a tour, including return to start node.
    (If start node is already added as last node in tour, 0 will be added to tour length.)
    :param nodes: all nodes in the graph in form of (x, y) coordinates.
    :param tour: list of node indices denoting a (potentially partial) tour.
    :return: tour length.
    """
    tour_length = 0
    for i in range(len(tour)):
        if i < len(tour) - 1:
            tour_length += np.linalg.norm(np.asarray(nodes[tour[i]]) - np.asarray(nodes[tour[i + 1]]))
        else:
            tour_length += np.linalg.norm(np.asarray(nodes[tour[-1]]) - np.asarray(nodes[tour[0]]))

    return tour_length


def cost(nodes: list[tuple], tour: list[int]) -> float:
    """
    Compute a tour cost.
    :param nodes: list of tuples of nodes.
    :param tour: list of nodes conforming a tour.
    :return: tour cost.
    """
    return -compute_tour_length(nodes, tour)


def compute_reward(nodes: list[tuple], old_state: list[int], state: list[int]):
    """
    Compute the current state reward based on the old state.
    :param nodes: list of tuples of nodes.
    :param old_state: the old state.
    :param state: the current state.
    :return: reward of the new state.
    """
    return cost(nodes, state) - cost(nodes, old_state)


def get_masks_for_actions(edge_weights: dict, partial_tours: list[list[int]]):
    """
    Compute weight mask for actions.
    :param edge_weights: dictionary of edge weights.
    :param partial_tours: list of partial tours.
    :return: weight mask array.
    """
    batch_masks = []
    for tour_ix, partial_tour in enumerate(partial_tours):
        mask = []
        for edge, weight in edge_weights[tour_ix].items():
            if edge in partial_tour or (edge[1], edge[0]) in partial_tour:
                mask.append(weight)
            else:
                mask.append(0)

        batch_masks.append(mask)

    return np.asarray(batch_masks)
