import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np


class Node:
    node_count = 1

    def __init__(self, x, y, name=None):
        self.x = x
        self.y = y
        if name is None:
            self.name = f'Node {Node.node_count}'
            Node.node_count += 1
        else:
            self.name = name


class Graph:
    def __init__(self, nodes):
        if len(nodes) < 2:
            raise ValueError('Cannot create graph: must have at least two nodes')
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if nodes[i].x == nodes[j].x and nodes[i].y == nodes[j].y:
                    raise ValueError('Cannot create graph: nodes must have unique coordinates')
        self.nodes = nodes


def calculate_manhattan_distance(node1, node2):
    return abs(node1.x - node2.x) + abs(node1.y - node2.y)


def manhattan_path(ax, node1, node2):
    ax.plot([node1.x, node1.x], [node1.y, node2.y], 'bo-')
    ax.plot([node1.x, node2.x], [node2.y, node2.y], 'bo-')
    ax.text(node1.x, node1.y, node1.name, fontsize=8)
    ax.text(node2.x, node2.y, node2.name, fontsize=8)


# Generate a list of nodes with random coordinates, without duplicates


nodes = []
while len(nodes) < 14:
    x, y = np.random.randint(1, 10), np.random.randint(1, 10)
    if not any(node.x == x and node.y == y for node in nodes):
        nodes.append(Node(x, y))
nodes.extend([Node(1, 1), Node(1, 7), Node(7, 1), Node(7, 7)])



# Create a new Graph object
graph = Graph(nodes)

# Initialize the tkinter window and canvas
root = tk.Tk()
root.geometry("800x800")
root.withdraw()

# Initialize variables to store the original and optimized distances
original_distance = 0
optimized_distance = 0

# Initialize the matplotlib figure and axes for the original and optimized configurations
fig = Figure(figsize=(8, 8), dpi=100)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Set the titles and limits for the axes
ax1.set_title("Original Configuration")
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)

ax2.set_title("Optimized Configuration")
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

# Draw the Manhattan paths between the nodes in the original configuration
for i in range(len(nodes) - 1):
    node1 = nodes[i]
    node2 = nodes[i + 1]
    manhattan_path(ax1, node1, node2)
    original_distance += calculate_manhattan_distance(node1, node2)

optimized_nodes = [nodes[0]]
remaining_nodes = nodes[1:]

while remaining_nodes:
    current_node = optimized_nodes[-1]
    closest_node = min(remaining_nodes, key=lambda x: calculate_manhattan_distance(current_node, x))
    optimized_nodes.append(closest_node)
    remaining_nodes.remove(closest_node)

for i in range(len(optimized_nodes) - 1):
    node1 = optimized_nodes[i]
    node2 = optimized_nodes[i + 1]
    manhattan_path(ax2, node1, node2)
    optimized_distance += calculate_manhattan_distance(node1, node2)

print("Original total distance:", original_distance)
print("Optimized total distance:", optimized_distance)

# Initialize the canvas and pack it into the tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Show the tkinter window and run the mainloop
root.deiconify()
root.mainloop()
