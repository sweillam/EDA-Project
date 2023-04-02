#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import itertools
import copy
import time
start_time = time.time()

class Node:
    # A class-level attribute to keep track of the total number of nodes created
    node_count = 1

    def __init__(self, x, y):
        # Initialize the node with x and y coordinates
        self.x = x
        self.y = y
        # Assign a name to the node using the current node_count value
        self.name = f'Node {Node.node_count}'
        # Increment the node_count each time a new node is created
        Node.node_count += 1

    def __eq__(self, other):
        # Check if the other object is an instance of the Node class
        if isinstance(other, Node):
            # Compare the x, y coordinates and name of the two nodes for equality
            return self.x == other.x and self.y == other.y and self.name == other.name
        # If the other object is not an instance of Node, return False
        return False

    def __hash__(self):
        # Define a hash function for the Node class based on the x, y coordinates and name
        return hash((self.x, self.y, self.name))

def count_crossing_edges(paths):
    # Initialize the count of crossing edges
    count = 0
    
    # Iterate through all pairs of paths
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            path1, path2 = paths[i], paths[j]
            
            # Check if the two paths cross each other
            if do_paths_cross(path1, path2):
                # Increment the count if the paths cross
                count += 1
                
    return count

def find_layers(paths):
    # Initialize the list of layers
    layers = []
    
    # Iterate through all paths
    for path in paths:
        inserted = False
        
        # Iterate through all layers
        for layer in layers:
            crosses_or_overlaps = False
            
            # Iterate through all paths in the current layer
            for layer_path in layer:
                # Check if the path crosses or overlaps the current layer_path
                if do_paths_cross(path, layer_path) or do_paths_cross(layer_path, path):
                    crosses_or_overlaps = True
                    break
            
            # If the path does not cross or overlap any paths in the current layer, add it to the layer
            if not crosses_or_overlaps:
                layer.append(path)
                inserted = True
                break
        
        # If the path is not inserted into any existing layers, create a new layer with the path
        if not inserted:
            layers.append([path])
    
    return len(layers), layers

def do_paths_cross(path1, path2):
    # Generate the list of segments for each path
    segments1 = [(path1[i], path1[i + 1]) for i in range(len(path1) - 1)]
    segments2 = [(path2[i], path2[i + 1]) for i in range(len(path2) - 1)]

    # Iterate through all segment pairs from the two paths
    for seg1 in segments1:
        for seg2 in segments2:
            
            # Check if seg1 is vertical and seg2 is horizontal
            if seg1[0].x == seg1[1].x and seg2[0].y == seg2[1].y:
                # Check if the segments cross each other
                if min(seg1[0].y, seg1[1].y) < seg2[0].y < max(seg1[0].y, seg1[1].y) and min(seg2[0].x, seg2[1].x) < seg1[0].x < max(seg2[0].x, seg2[1].x):
                    return True
            
            # Check if seg1 is horizontal and seg2 is vertical
            elif seg1[0].y == seg1[1].y and seg2[0].x == seg2[1].x:
                # Check if the segments cross each other
                if min(seg1[0].x, seg1[1].x) < seg2[0].x < max(seg1[0].x, seg1[1].x) and min(seg2[0].y, seg2[1].y) < seg1[0].y < max(seg2[0].y, seg2[1].y):
                    return True
            
            # Check if both seg1 and seg2 are vertical and overlapping
            elif seg1[0].x == seg1[1].x == seg2[0].x == seg2[1].x:
                # Check if the segments overlap
                if min(seg1[0].y, seg1[1].y) < min(seg2[0].y, seg2[1].y) < max(seg1[0].y, seg1[1].y) or min(seg1[0].y, seg1[1].y) < max(seg2[0].y, seg2[1].y) < max(seg1[0].y, seg1[1].y):
                    return True
            
            # Check if both seg1 and seg2 are horizontal and overlapping
            elif seg1[0].y == seg1[1].y == seg2[0].y == seg2[1].y:
                # Check if the segments overlap
                if min(seg1[0].x, seg1[1].x) < min(seg2[0].x, seg2[1].x) < max(seg1[0].x, seg1[1].x) or min(seg1[0].x, seg1[1].x) < max(seg2[0].x, seg2[1].x) < max(seg1[0].x, seg1[1].x):
                    return True
                    
    # If no crossing segments are found, return False
    return False

def manhattan_path_with_layer_limit(node1, node2, paths, max_layers):
    # Calculate the differences in x and y coordinates between node1 and node2
    x_diff = node2.x - node1.x
    y_diff = node2.y - node1.y

    # If the nodes are aligned along either the x-axis or y-axis, add the direct path between them
    if x_diff == 0 or y_diff == 0:
        paths.append([node1, node2])
        return

    # Generate candidate paths for connecting node1 and node2 using the Manhattan method
    candidate_paths = []
    candidate_paths.append([node1, Node(node1.x, node2.y), node2])
    candidate_paths.append([node1, Node(node2.x, node1.y), node2])
    candidate_paths.append([node2, Node(node1.x, node2.y), node1])  # Reversed path
    candidate_paths.append([node2, Node(node2.x, node1.y), node1])  # Reversed path

    # Initialize variables to keep track of the best path found so far
    min_crossings = float('inf')
    best_path = None

    # Iterate through the candidate paths and determine the one with the least number of crossings
    for path in candidate_paths:
        # Create a temporary copy of the paths list with the current candidate path added
        temp_paths = paths.copy()
        temp_paths.append(path)
        
        # Calculate the number of crossing edges and layers for the temporary paths list
        crossing_edges_count = count_crossing_edges(temp_paths)
        layers_count, _ = find_layers(temp_paths)

        # If a path with no crossings is found, choose it as the best path and break the loop
        if crossing_edges_count == 0:
            best_path = path
            break
        # If the path has fewer crossings than the current minimum and doesn't exceed the maximum layers, choose it as the best path
        elif layers_count <= max_layers and crossing_edges_count < min_crossings:
            best_path = path
            min_crossings = crossing_edges_count

    # If a best path is found, add it to the paths list
    if best_path is not None:
        paths.append(best_path)
    else:
        # If none of the candidate paths have less crossings, choose the one with minimum layers
        min_layers = float('inf')
        best_path = None
        for path in candidate_paths:
            temp_paths = paths.copy()
            temp_paths.append(path)
            layers_count, _ = find_layers(temp_paths)
            if layers_count < min_layers:
                best_path = path
                min_layers = layers_count
        paths.append(best_path)
        
def compute_total_path_length(paths):
    # Initialize the total path length to zero
    total_length = 0
    
    # Iterate through all paths
    for path in paths:
        # Iterate through all pairs of consecutive nodes in the path
        for i in range(len(path) - 1):
            # Compute the distance between the two nodes and add it to the total length
            total_length += np.sqrt((path[i+1].x - path[i].x)**2 + (path[i+1].y - path[i].y)**2)
    
    return total_length

#input nodes and connections 
#put connections and nodes here 
def get_user_input(number_of_nodes,samples, number_of_connections):
    #test cases
    nodes = []
    connections = []
    while len(nodes) < number_of_nodes:
        x, y = np.random.randint(1, samples), np.random.randint(1, samples)
        if not any(node.x == x and node.y == y for node in nodes):
            nodes.append(Node(x, y))
    while len(connections) < number_of_connections:
        i, j = np.random.randint(0, number_of_nodes), np.random.randint(0, number_of_nodes)
        if i != j:
            connections.append((i,j))
    return nodes, connections

layer_colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']

fig = plt.figure(figsize=(5, 5), dpi=100)
ax = fig.add_subplot(111)

number_of_nodes = 10
samples = 100
number_of_connections = 13
nodes, connections = get_user_input(number_of_nodes,samples,number_of_connections)

#maximum layers 
max_layers = 10

# Create the initial_paths list which contains the direct connections between nodes in connections
initial_paths = [[nodes[i], nodes[j]] for i, j in connections]
            
# Create an empty list to store the paths after applying the manhattan_path_with_layer_limit function
paths = []

# Iterate through the connections
for i, j in connections:
    # Get the two nodes that need to be connected
    node1, node2 = nodes[i], nodes[j]
     # Compute the Manhattan path with the layer limit and store it in the paths list
    manhattan_path_with_layer_limit(node1, node2, paths, max_layers)

# Find the total number of layers and the subset of paths in each layer
layers_count, max_layer_subset = find_layers(paths)

# Iterate through the nodes and plot them on the graph as blue dots
for node in nodes:
    ax.scatter(node.x, node.y, color='blue')
    # Add the node name as a label next to each dot
    ax.text(node.x + 0.1, node.y + 0.1, node.name, fontsize=8)

# Create a dictionary to store the layer index for each path
path_to_layer = {}
# Iterate through the layers and their paths
for i, layer in enumerate(max_layer_subset):
    # Assign the layer index to each path in the layer
    for path in layer:
        path_to_layer[tuple(map(id, path))] = i % len(layer_colors)

total_length = compute_total_path_length(paths)
print('Total length is:',total_length)

# Iterate through the paths and plot them on the graph
for path in paths:
    # Iterate through the segments in the path
    for i in range(len(path) - 1):
        # Create a unique identifier for the path based on the node IDs
        path_key = tuple(map(id, path))
        # Check if the path_key is present in path_to_layer dictionary
        if path_key in path_to_layer:
            # Get the color index from the path_to_layer dictionary
            color_index = path_to_layer[path_key]
            # Plot the segment with the corresponding color from the layer_colors list
            ax.plot([path[i].x, path[i + 1].x], [path[i].y, path[i + 1].y], color=layer_colors[color_index])
        else:
            # If the path_key is not found in the path_to_layer dictionary, plot the segment with a 'gray' color
            ax.plot([path[i].x, path[i + 1].x], [path[i].y, path[i + 1].y], color='gray')

#axis grids
ax.grid(True)
ax.set_xlim(np.min([node.x for node in nodes]) - 1, np.max([node.x for node in nodes]) + 1)
ax.set_ylim(np.min([node.y for node in nodes]) - 1, np.max([node.y for node in nodes]) + 1)
ax.set_ylabel('y')
ax.set_xlabel('x')

print('Number of layers:',layers_count)
print("--- %s seconds ---" % (time.time() - start_time))
fig.show()

