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

        self.neighbors = []

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

def calculate_manhattan_distance(node1, node2):
    return abs(node1.x - node2.x) + abs(node1.y - node2.y)

def manhattan_path(ax, node1, node2):
    ax.plot([node1.x, node1.x], [node1.y, node2.y], 'bo-')
    ax.plot([node1.x, node2.x], [node2.y, node2.y], 'bo-')
    ax.text(node1.x, node1.y, node1.name, fontsize=8)
    ax.text(node2.x, node2.y, node2.name, fontsize=8)

root = tk.Tk()
root.geometry("800x800")
root.withdraw()

nodes = [Node(1, 1), Node(1, 7), Node(7, 1), Node(7, 7)]
for i in range(10):
    nodes.append(Node(np.random.randint(1, 10), np.random.randint(1, 10)))

for node in nodes:
    for other_node in nodes:
        if node != other_node:
            node.add_neighbor(other_node)

original_distance = 0
optimized_distance = 0

fig = Figure(figsize=(8, 8), dpi=100)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.set_title("Original Configuration")
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)

ax2.set_title("Optimized Configuration")
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

for i in range(len(nodes) - 1):
    node1 = nodes[i]
    node2 = nodes[i + 1]
    manhattan_path(ax1, node1, node2)
    original_distance += calculate_manhattan_distance(node1, node2)

optimized_nodes = [nodes[0]]
remaining_nodes = nodes[1:]

while remaining_nodes:
    current_node = optimized_nodes[-1]
    min_distance = float('inf')
    min_neighbor = None

    for neighbor in current_node.neighbors:
        if neighbor in remaining_nodes:
            distance = calculate_manhattan_distance(current_node, neighbor)
            if distance < min_distance:
                min_distance = distance
                min_neighbor = neighbor

    optimized_nodes.append(min_neighbor)
    remaining_nodes.remove(min_neighbor)

for i in range(len(optimized_nodes) - 1):
    node1 = optimized_nodes[i]
    node2 = optimized_nodes[i + 1]
    manhattan_path(ax2, node1, node2)
    optimized_distance += calculate_manhattan_distance(node1, node2)

print("Original total distance:", original_distance)
print("Optimized total distance:", optimized_distance)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

root.deiconify()
root.mainloop()
