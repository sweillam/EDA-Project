import tkinter as tk
from tkinter import simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

class Node:
    node_count = 1

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.name = f'Node {Node.node_count}'
        Node.node_count += 1
        self.neighbours = []

    def add_neighbour(self, neighbour):
        self.neighbours.append(neighbour)

    def __hash__(self):
        return hash((self.x, self.y, self.name))

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.x == other.x and self.y == other.y and self.name == other.name
        return False

def heuristic(node1, node2):
    return abs(node1.x - node2.x) + abs(node1.y - node2.y)

def get_corner_points(node1, node2):
    x_diff = node2.x - node1.x
    y_diff = node2.y - node1.y
    candidates = []
    if x_diff != 0 and y_diff != 0:
        candidates += [Node(node1.x + x_diff, node1.y),
                       Node(node1.x + x_diff, node2.y),
                       Node(node1.x, node1.y + y_diff),
                       Node(node2.x, node1.y + y_diff)]
    elif x_diff == 0:
        candidates += [Node(node1.x, node1.y + y_diff)]
    elif y_diff == 0:
        candidates += [Node(node1.x + x_diff, node1.y)]
    return candidates


def astar(start, end, nodes, existing_paths):
    def is_path_crossing(existing_paths, new_path):
        def is_line_intersecting(a, b, c, d):
            def ccw(p, q, r):
                return (r.y - p.y) * (q.x - p.x) > (q.y - p.y) * (r.x - p.x)

            return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

        for existing_path_key, existing_path in existing_paths.items():
            for i in range(len(existing_path) - 1):
                for j in range(len(new_path) - 1):
                    if is_line_intersecting(existing_path[i], existing_path[i + 1], new_path[j], new_path[j + 1]):
                        return True
        return False

    open_set = [start]
    closed_set = set()
    g_score = {node: float('inf') for node in nodes}
    f_score = {node: float('inf') for node in nodes}
    g_score[start] = 0
    f_score[start] = heuristic(start, end)
    parent = {}

    while open_set:
        current = min(open_set, key=lambda x: f_score[x])
        if current == end:
            # reconstruct the path
            path = [end]
            while path[-1] != start:
                parent_node, corner_point = parent[path[-1]]
                if corner_point is not None:
                    path.append(corner_point)
                path.append(parent_node)
            path.reverse()

            if is_path_crossing(existing_paths, path):
                return None

            return path

        open_set.remove(current)
        closed_set.add(current)

        for neighbour in current.neighbours:
            if neighbour in closed_set:
                continue

            corner_points = get_corner_points(current, neighbour)
            tentative_g_scores = [g_score[current] + heuristic(current, cp) + heuristic(cp, neighbour) for cp in corner_points]
            min_tentative_g_score = min(tentative_g_scores)
            min_corner_point = corner_points[np.argmin(tentative_g_scores)]

            if neighbour not in open_set:
                open_set.append(neighbour)
            elif min_tentative_g_score >= g_score[neighbour]:
                continue

            parent[neighbour] = (current, min_corner_point if len(corner_points) > 1 else None)
            g_score[neighbour] = min_tentative_g_score
            f_score[neighbour] = g_score[neighbour] + heuristic(neighbour, end)

    return None

def get_user_input():
    num_nodes = 4
    nodes = [Node(1, 1), Node(5, 0), Node(3, 3), Node(3, 5)]

    nodes[0].add_neighbour(nodes[1])
    nodes[0].add_neighbour(nodes[2])
    nodes[1].add_neighbour(nodes[2])
    nodes[1].add_neighbour(nodes[3])
    nodes[2].add_neighbour(nodes[3])
    nodes[2].add_neighbour(nodes[0])
    nodes[2].add_neighbour(nodes[1])
    nodes[3].add_neighbour(nodes[2])

    return nodes

root = tk.Tk()
root.title("Manhattan Shortest Paths")
root.geometry("600x600")

fig = Figure(figsize=(5, 5), dpi=100)
ax = fig.add_subplot(111)

nodes = get_user_input()

paths = {}
for node1 in nodes:
    for node2 in nodes:
        if node1 == node2:
            continue

        path = astar(node1, node2, nodes, paths)
        if path:
            paths[(node1, node2)] = path
            paths[(node2, node1)] = list(reversed(path))  # Add the reversed path as well

for node in nodes:
    ax.scatter(node.x, node.y, color='blue')
    ax.text(node.x + 0.1, node.y + 0.1, node.name, fontsize=8)

for (node1, node2), path in paths.items():
    path = [p for p in path if isinstance(p, Node)]
    for i in range(len(path) - 1):
        ax.plot([path[i].x, path[i + 1].x], [path[i].y, path[i + 1].y], color='red')

ax.grid(True)
ax.set_xlim(np.min([node.x for node in nodes]) - 1, np.max([node.x for node in nodes]) + 1)
ax.set_ylim(np.min([node.y for node in nodes]) - 1, np.max([node.y for node in nodes]) + 1)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

tk.mainloop()
