from app.agent.graph import agent_graph


def test_agent_graph_contains_connected_flow_nodes() -> None:
    graph = agent_graph.get_graph()
    node_names = set(graph.nodes)

    assert {
        "__start__",
        "nlu",
        "merge_context_initial",
        "reference",
        "merge_context_after_reference",
        "planner",
        "ask",
        "tool",
        "apply_result",
        "response",
        "__end__",
    }.issubset(node_names)

    edge_pairs = {(edge.source, edge.target) for edge in graph.edges}
    assert ("nlu", "merge_context_initial") in edge_pairs
    assert ("merge_context_initial", "reference") in edge_pairs
    assert ("reference", "merge_context_after_reference") in edge_pairs
    assert ("merge_context_after_reference", "planner") in edge_pairs
    assert ("apply_result", "response") in edge_pairs
    assert ("response", "__end__") in edge_pairs
