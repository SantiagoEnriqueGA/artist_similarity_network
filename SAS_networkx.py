import csv
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import json 


class Graph:
    def __init__(self, with_nodes_file=None, with_edges_file=None):
        """
        option 1:  init as an empty graph and add nodes
        option 2: init by specifying a path to nodes & edges files
        """
        self.edges = []
        self.nodes = defaultdict(str)
        self.nodes_sim = defaultdict(str)
        self.g = nx.Graph()

        if with_nodes_file and with_edges_file:
            nodes_CSV = csv.reader(open(with_nodes_file))
            nodes_CSV = list(nodes_CSV)[1:]
            self.nodes = [(n[0], n[1]) for n in nodes_CSV]

            edges_CSV = csv.reader(open(with_edges_file))
            edges_CSV = list(edges_CSV)[1:]
            self.edges = [(e[0], e[1]) for e in edges_CSV]

    def add_node(self, id: str, name: str) -> None:
        """
        add a tuple (id, name) representing a node to self.nodes if it does not already exist
        The graph should not contain any duplicate nodes
        """
        self.nodes[id] = name

        return self.nodes

    def add_node_sim(self, id: str, name: str, num_sim: int) -> None:
        """
        add a tuple (id, name) representing a node to self.nodes if it does not already exist
        The graph should not contain any duplicate nodes
        """      
        self.nodes_sim[id] = name.split()[:num_sim]

        return self.nodes_sim

    def add_edge(self, source: str, target: str) -> None:
        """
        Add an edge between two nodes if it does not already exist.
        An edge is represented by a tuple containing two strings: e.g.: ('source', 'target').
        Where 'source' is the id of the source node and 'target' is the id of the target node
        e.g., for two nodes with ids 'a' and 'b' respectively, add the tuple ('a', 'b') to self.edges
        """
        edge = (source, target)
        reversed_edge = (target, source)

        if edge not in self.edges and reversed_edge not in self.edges:
            self.edges.append(edge)

        return self.edges

    def total_nodes(self) -> int:
        """
        Returns an integer value for the total number of nodes in the graph
        """
        return len(self.nodes)

    def total_edges(self) -> int:
        """
        Returns an integer value for the total number of edges in the graph
        """
        return len(self.edges)

    def max_degree_nodes(self) -> dict:
        """
        Return the node(s) with the highest degree
        Return multiple nodes in the event of a tie
        Format is a dict where the key is the node_id and the value is an integer for the node degree
        e.g. {'a': 8}
        or {'a': 22, 'b': 22}
        """
        # Create a dictionary to store the degree of each node
        node_degree = defaultdict(int)

        # Calculate the degree for each node
        for edge in self.edges:
            node_degree[edge[0]] += 1  # Increment the degree for the source node
            node_degree[edge[1]] += 1  # Increment the degree for the target node

        # Find the maximum degree
        max_degree = max(node_degree.values())

        # Filter nodes with the maximum degree
        max_degree_nodes = {node: degree for node, degree in node_degree.items() if degree == max_degree}

        return max_degree_nodes


    def print_nodes(self):
        """
        No further implementation required
        May be used for de-bugging if necessary
        """
        print(self.nodes)

    def print_nodes_sim(self):
        """
        No further implementation required
        May be used for de-bugging if necessary
        """
        print(self.nodes_sim)

    def print_edges(self):
        """
        No further implementation required
        May be used for de-bugging if necessary
        """
        print(self.edges)

    # Do not modify
    def write_edges_file(self, path="network_data/edges.csv")->None:
        """
        write all edges out as .csv
        :param path: string
        :return: None
        """
        edges_path = path
        edges_file = open(edges_path, 'w', encoding='utf-8')

        edges_file.write("source" + "," + "target" + "\n")

        for e in self.edges:
            edges_file.write(e[0] + "," + e[1] + "\n")

        edges_file.close()
        print("finished writing edges to csv")

    # Do not modify
    def write_nodes_file(self, path="network_data/nodes.json")->None:
        """
        write all nodes out as .csv
        :param path: string
        :return: None
        """
        # Convert and write JSON object to file
        with open(path, "w") as outfile: 
            json.dump(self.nodes, outfile)

        print("finished writing nodes to json")


    def parse_artist_txt(self, filename: str):
        """"
        gets all artists and ids from txt
        and adds to node
        """
        with open(filename) as tsv:
            for line in csv.reader(tsv, dialect="excel-tab"):
                self.add_node(line[0],line[1])
        return  
    
    def parse_sim_artist_txt(self,filename: str, node_sim: id):
        """"
        gets all ids and similar ids from txt
        and adds to node_sim
        """
        with open(filename) as tsv:
            for line in csv.reader(tsv, dialect="excel-tab"):
                self.add_node_sim(line[0],line[1], node_sim)
        return
    
    def node_get_id_from_name(self,name: str):
        return list(graph.nodes.keys())[list(graph.nodes.values()).index(name)]
    
    def add_networkx_node(self, one: str):
        """"
        gets node id, writes name
        """
        self.g.add_node(self.nodes[one])
        return

    def add_networkx_edge(self, one: str, two: str):
        """"
        gets node ids, writes names
        """
        self.g.add_edge(self.nodes[one], self.nodes[two])
        return
    
    def add_networkx_depth(self, id: str, loops: int):
        """"
        adds nodes and edges, by depth
        """
        # get sim artists
        sim = self.nodes_sim[id]
        # for each similar
        for s in sim:
            # add node and edge between nodes
            self.add_networkx_node(s)
            self.add_networkx_edge(id, s)

            if loops > 0:
                self.add_networkx_depth(s, loops-1)        
        return
    
    def draw_networkx_node(self, lab:bool):
        """
        draws node graph to file
        """
        # draw_circular(G, keywords) : This gives circular layout of the graph G.
        # draw_planar(G, keywords) :] This gives a planar layout of a planar networkx graph G.
        # draw_random(G, keywords) : This gives a random layout of the graph G.
        # draw_spectral(G, keywords) : This gives a spectral 2D layout of the graph G.
        # draw_spring(G, keywords) : This gives a spring layout of the graph G.
        # draw_shell(G, keywords) : This gives a shell layout of the graph G. 

        # nx.draw(self.g, with_labels = lab)
        
        # smaller nodes and fonts
        # plt.figure(2)
        # nx.draw(self.g,node_size=60,font_size=8) 

        # larger figure size
        plt.figure(figsize=(12,16)) 
        nx.draw(self.g,node_size=30,font_size=6,with_labels=lab)
        plt.savefig("artists.png")
        plt.show()

        return



if __name__ == '__main__':
    graph = Graph()
    
    num_similar = 10

    graph.parse_artist_txt("dataset-artist-similarity/LastFM/mb2uri_lastfmapi.txt")
    graph.parse_sim_artist_txt("dataset-artist-similarity/LastFM/lastfmapi_gold.txt", num_similar)

    # graph.print_nodes()
    # graph.print_edges()
    # print(graph.total_nodes())
    # graph.print_nodes_sim()
    # graph.write_edges_file()
    # graph.write_nodes_file()

    ### BUILD GRAPH AROUND ARTIST
    # # get id of artist
    # id = graph.node_get_id_from_name('switchfoot')
    # # add original node
    # graph.add_networkx_node(id)
    # # add network nodes/edges
    # graph.add_networkx_depth(id,6)
    # graph.draw_networkx_node(True)

    for x in graph.nodes_sim:
        graph.add_networkx_node(x)
        for sim in graph.nodes_sim[x]:
            graph.add_networkx_edge(x, sim)

    print(f'Number of nodes: {len(graph.g.nodes())}')
    print(f'Number of edges: {len(graph.g.edges())}')

    artist_1 = 'switchfoot'
    artist_2 = 'taylor swift'

    try:
        path = nx.shortest_path(graph.g, source=artist_1, target=artist_2)
        print(f'Shortest path between {artist_1} and {artist_2}:')
        print(path)

        result_graph = graph.g.subgraph(path)
        nx.draw(result_graph, font_weight='bold', with_labels=True)
        plt.savefig(f'plots/{artist_1}_to_{artist_2}.png')
        plt.show()

        plt.figure(figsize=(14,6))
        pos = nx.spring_layout(graph.g, seed=1234)
        nx.draw(graph.g, pos, node_color='k', node_size=3, with_labels=False)

        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(graph.g, pos, nodelist=path, node_color='r', node_size=50)
        nx.draw_networkx_edges(graph.g, pos, edgelist=path_edges, edge_color='r', width=2)
        
        labels = {}    
        for node in graph.g.nodes():
            if node in path:
                #set the node name as the key and the label as its value 
                labels[node] = node
        nx.draw_networkx_labels(graph.g,pos,labels,font_size=12,font_color='r')
        
        plt.savefig(f'plots/{artist_1}_to_{artist_2}_fullNetwork.png')
        plt.show()

    except nx.NetworkXNoPath:
        print(f"No path between {artist_1} and {artist_2}")
    except nx.NodeNotFound as e:
        print(e)
    


    



    
    