import networkx as nx


class NetworkBuilder:

    """
    Converts detected nodes and edges
    into a NetworkX graph.
    """

    ##########################################################

    @staticmethod
    def build(points, labels, edges):

        G = nx.Graph()

        # ----------------------------
        # Add Nodes
        # ----------------------------

        for label, point in zip(labels, points):

            G.add_node(

                label,

                pos=point

            )

        # ----------------------------
        # Node Lookup
        # ----------------------------

        lookup = {}

        for label, point in zip(labels, points):

            lookup[point] = label

        # ----------------------------
        # Add Edges
        # ----------------------------

        for edge in edges:

            start = lookup[edge["start"]]

            end = lookup[edge["end"]]

            G.add_edge(

                start,

                end,

                weight=edge["length"]

            )

        return G