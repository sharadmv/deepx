from node import CompositeNode, ConcatenateNode
import networkx as nx

def draw(model):
    graph = nx.DiGraph()

    populate_graph(model, graph)

    layout = nx.spring_layout(graph)

    nodes = graph.nodes()
    labels = {g:label(g) for g in nodes}

    nx.draw_networkx_nodes(graph, layout,
                       nodelist=nodes,
                       node_color='b',
                       node_size=80)

    nx.draw_networkx_edges(graph, layout,
                       edgelist=graph.edges(),
                       width=1,alpha=0.5,edge_color='black')

    nx.draw_networkx_labels(graph, layout,
                            labels=labels,
                            fontsize=16)

def label(node):
    if isinstance(node, CompositeNode):
        return
    elif isinstance(node, ConcatenateNode):
        return '+'
    return str(node)

def connect(in_node, out_node, graph):
    if isinstance(in_node, CompositeNode):
        connect(in_node.out_node, out_node, graph)
        return
    elif isinstance(out_node, CompositeNode):
        connect(in_node, out_node.in_node)
        return
    graph.add_edge(in_node, out_node)

def populate_graph(node, graph):
    if isinstance(node, CompositeNode):
        populate_graph(node.in_node, graph)
        populate_graph(node.out_node, graph)
        connect(node.in_node, node.out_node, graph)
        return
    elif isinstance(node, ConcatenateNode):
        populate_graph(node.left_node, graph)
        populate_graph(node.right_node, graph)
        graph.add_node(node)
        connect(node.left_node, node, graph)
        connect(node.right_node, node, graph)
        return
    graph.add_node(node)
