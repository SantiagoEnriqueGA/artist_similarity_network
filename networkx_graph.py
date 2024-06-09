import csv
from collections import deque, defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import json 
import plotly.graph_objects as go
import plotly.express as px
import community as community_louvain 

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

    def find_nodes_with_furthest_connectivity(self):
        """
        Find nodes with the furthest connectivity.

        This function calculates the nodes with the most significant number of connections 
        between them on the shortest path.

        Returns:
        list: List of nodes with furthest connectivity.
        """
        max_connectivity = 0
        furthest_nodes = []

        # Perform BFS starting from each node to find shortest paths
        for node in self.g.nodes():
            visited = {node}
            queue = deque([(node, 0)])  # Initialize queue with the node and its distance
            while queue:
                current, distance = queue.popleft()
                for neighbor in self.g.neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
                        # Check if the current distance is the maximum connectivity found so far
                        if distance + 1 > max_connectivity:
                            max_connectivity = distance + 1
                            furthest_nodes = [(node, neighbor)]  # Start new list
                        elif distance + 1 == max_connectivity:
                            furthest_nodes.append((node, neighbor))  # Add to existing list

        return furthest_nodes, max_connectivity
    
    def draw_plotly_network(self, path=None):
        """
        Draws an interactive plot of the network using Plotly.

        This function uses Plotly to create an interactive visualization of the network graph.
        It displays nodes and edges with a spring layout, allowing for zooming and panning.

        Parameters:
        path (list): Optional. List of nodes representing the shortest path to highlight.
        """
        # Calculate node positions using a spring layout
        pos = nx.spring_layout(self.g, seed=1234)
        
        # Initialize lists to hold edge coordinates
        edge_x = []
        edge_y = []
        path_edge_x = []
        path_edge_y = []

        if path:
            artist_1 = path[0]
            artist_2 = path[-1]

        # Create a set of edges in the path for easy lookup
        path_edges = set(zip(path, path[1:])) if path else set()

        # Extract the x and y coordinates for edges
        for edge in self.g.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            if edge in path_edges or (edge[1], edge[0]) in path_edges:
                path_edge_x.extend([x0, x1, None])
                path_edge_y.extend([y0, y1, None])
            else:
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        # Create a Plotly scatter trace for non-path edges
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Create a Plotly scatter trace for path edges
        path_edge_trace = go.Scatter(
            x=path_edge_x, y=path_edge_y,
            line=dict(width=2, color='red'),  # Increase width for path edges
            hoverinfo='none',
            mode='lines'
        )

        # Initialize lists to hold node coordinates and text
        node_x = []
        node_y = []
        node_text = []
        node_text_hover = []
        node_color = []
        node_size = []

        # Set default color and highlight path nodes
        for node in self.g.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            if path and node in path:
                node_text.append(node)  # Add label only for path nodes
                node_text_hover.append(node)  # Add label only for path nodes
                node_color.append('red')
                node_size.append(15)  # Increase size for path nodes
            else:
                node_text.append('')  # No label for other nodes
                node_text_hover.append(node)  # Add label only for path nodes
                node_color.append('black')
                node_size.append(7.5)  # Default size for other nodes

        # Create a Plotly scatter trace for nodes
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',  # Add text mode to display labels
            hoverinfo='text',
            marker=dict(
                showscale=False,  # Show color scale
                colorscale='YlGnBu',  # Color scale
                color=node_color,  # Set node color
                size=node_size,  # Set node size
                line=dict(width=0),  # Remove white outline around nodes
            ),
            text=node_text,  # Text labels for nodes
            hovertext=node_text_hover,  # Hover text labels for nodes
            textposition='top center',  # Position labels at the top center
            textfont=dict(color='red')  # Color labels red
        )

        # Create a Plotly figure with the node and edge traces
        if path: subtitle=f"Shortest path from {artist_1} to {artist_2}"
        else:    subtitle=""
        fig = go.Figure(data=[edge_trace, path_edge_trace, node_trace],
                        layout=go.Layout(
                            title='Artist Similarity Network',  # Title of the graph
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),  # Margins
                            annotations=[dict(
                                text=subtitle,
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002
                            )],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),  # Hide x-axis grid, line, and ticks
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)   # Hide y-axis grid, line, and ticks
                        )
        )

        # Display the interactive Plotly figure
        fig.show()

    def draw_plotly_network_cluster(self):
        """
        Draws an interactive plot of the network using Plotly. Clusters each point and colors accordingly.

        This function uses Plotly to create an interactive visualization of the network graph.
        It displays nodes and edges with a spring layout, allowing for zooming and panning.
        """
        
        # Calculate node positions using a spring layout
        pos = nx.spring_layout(self.g, seed=1234)
        
        # Initialize lists to hold edge coordinates
        edge_x = []
        edge_y = []

        # Extract the x and y coordinates for edges
        for edge in self.g.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Create a Plotly scatter trace for edges
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Detect communities using the Louvain method
        partition = community_louvain.best_partition(self.g)
        
        # Assign a unique color to each community
        unique_communities = set(partition.values())
        num_communities = len(unique_communities)
        color_palette = px.colors.qualitative.Alphabet * (num_communities // len(px.colors.qualitative.Alphabet) + 1)
        color_palette = color_palette[:num_communities]

        # Create a Plotly figure with the edge trace
        fig = go.Figure(data=[edge_trace],
                        layout=go.Layout(
                            title='Artist Similarity Network (Clustered)',  # Title of the graph
                            titlefont_size=16,
                            showlegend=True,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),  # Margins
                            annotations=[dict(
                                text='Clustered Graph',
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002
                            )],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),  # Hide x-axis grid, line, and ticks
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)   # Hide y-axis grid, line, and ticks
                        )
        )

        # Create a separate node trace for each community
        for community in unique_communities:
            community_node_x = []
            community_node_y = []
            community_node_text = []
            community_node_text_hover = []

            for node, comm in partition.items():
                if comm == community:
                    x, y = pos[node]
                    community_node_x.append(x)
                    community_node_y.append(y)
                    community_node_text.append('')  # No label
                    community_node_text_hover.append(node)  # Add label hover only

            # Create a Plotly scatter trace for nodes in this community
            community_node_trace = go.Scatter(
                x=community_node_x, y=community_node_y,
                mode='markers+text',  # Add text mode to display labels
                hoverinfo='text',
                marker=dict(
                    showscale=False,  # Show color scale
                    color=color_palette[community],  # Set community color
                    size=7.5,  # Set node size
                    line=dict(width=0),  # Remove white outline around nodes
                ),
                text=community_node_text,  # Text labels for nodes
                hovertext=community_node_text_hover,  # Hover text labels for nodes
                textposition='top center',  # Position labels at the top center
                textfont=dict(color='red'),  # Color labels red
                name=f'Community {community}'
            )

            # Add the community node trace to the figure
            fig.add_trace(community_node_trace)

        # Display the interactive Plotly figure
        fig.show()

if __name__ == '__main__':
    graph = Graph()
    
    num_similar = 5

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
    
    # Finds longest connections
    furthest_nodes, max_connectivity = graph.find_nodes_with_furthest_connectivity()
    print(f"Furthest nodes: {furthest_nodes}")
    print(f"Max connectivity: {max_connectivity}")

    artist_1 = 'e-40'
    artist_2 = 'adnan sami'

    # Draws whole network
    graph.draw_plotly_network()
    
    # Draws whole network, clustered
    graph.draw_plotly_network_cluster()


    try:
        path = nx.shortest_path(graph.g, source=artist_1, target=artist_2)
        print(f'Shortest path between {artist_1} and {artist_2}:')
        print(path)

        # Plot just path
        result_graph = graph.g.subgraph(path)
        nx.draw(result_graph, font_weight='bold', with_labels=True)
        plt.savefig(f'plots/{artist_1}_to_{artist_2}.png')
        plt.show()

        # Plot path on network
        plt.figure(figsize=(14,6))
        pos = nx.spring_layout(graph.g, seed=123)
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

        # Plotly visualization path on network
        graph.draw_plotly_network(path=path)
        
    except nx.NetworkXNoPath:
        print(f"No path between {artist_1} and {artist_2}")
    except nx.NodeNotFound as e:
        print(e)
