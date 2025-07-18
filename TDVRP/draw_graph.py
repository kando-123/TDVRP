import sys
import json
import matplotlib.pyplot as plot
import os
from matplotlib.patches import Circle
from math import hypot, sin, cos
from math import atan2 as phase

RADIUS = 1.4
DELTA = 0.2

def draw_vertex(axes, x, y, index, quantity, is_depot=False):
    circle = Circle((x, y), RADIUS, color='lightcoral' if is_depot else 'white', ec='black', linewidth=2, zorder=2)
    axes.add_patch(circle)
    if quantity > 0:
        axes.plot([x - RADIUS, x + RADIUS], [y, y], color='black', lw=1, zorder=3)
        axes.text(x, y + RADIUS * 0.4, str(index), ha='center', va='center', fontsize=8, color='black', zorder=4)
        axes.text(x, y - RADIUS * 0.6, str(quantity), ha='center', va='center', fontsize=8, color='red', zorder=4)
    else:
        axes.text(x, y, str(index), ha='center', va='center', fontsize=10, zorder=4, color='black')

def draw_edge(axes, x1, y1, x2, y2, color='gray'):
    (x, y) = (x2 - x1, y2 - y1)
    φ = phase(y, x)
    (δx, δy) = (DELTA * -sin(φ), DELTA * cos(φ))
    r = hypot(x, y)
    (Δx, Δy) = (x * RADIUS / r, y * RADIUS / r)
    axes.annotate('', (x2 - Δx + δx, y2 - Δy + δy), (x1 + Δx + δx, y1 + Δy + δy),
                  arrowprops=dict(arrowstyle='->', color=color, lw=2),
                  zorder=1)

def interpolate_color(value, min, max):
    if max == min:
        return (0.5, 0.5, 0.5)   # gray if all equal
    ratio = (value - min) / (max - min)
    return (ratio, 1 - ratio, 0)   # linear interpolation from green to red

def draw_graph(in_file, out_dir):
    with open(in_file) as f:
        graph = json.load(f)

    vertices = graph['vertices']
    edges = graph['edges']
    vertex_map = { v['i']: v for v in vertices }
    depot = graph.get('depot', 0)

    weights_lists = [ e['w'] for e in edges if 'w' in e ]
    if not weights_lists:
        print("No weighted edges found.")
        return

    count_weights = len(weights_lists[0])
    all_weights = [w for weights in weights_lists for w in weights]
    w_min = min(all_weights)
    w_max = max(all_weights)

    for index in range(count_weights):
        figure, axes = plot.subplots()
        axes.set_aspect('equal')

        for e in edges:
            (u, v) = (e['u'], e['v'])
            (x1, y1) = (vertex_map[u]['x'], vertex_map[u]['y'])
            (x2, y2) = (vertex_map[v]['x'], vertex_map[v]['y'])
            weight = e['w'][index] if 'w' in e else None
            color = interpolate_color(weight, w_min, w_max) if weight is not None else 'gray'
            draw_edge(axes, x1, y1, x2, y2, color)

        for v in vertices:
            i = v['i']
            (x, y) = (v['x'], v['y'])
            q = v.get('q', 0)
            draw_vertex(axes, x, y, i, q, is_depot=(i == depot))

        axes.autoscale()
        plot.xlabel('X')
        plot.ylabel('Y')
        plot.title(f'Graph Visualization - Weight index {index}')
        plot.grid(True)
        plot.axis('equal')
        
        path = f'graph_{index}.png'
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, path)   # prepend out_dir
        plot.savefig(path)
        
        plot.close()

if len(sys.argv) < 2:
    print("Usage: python draw_graph.py <input_graph.json> <output_directory>")
else:
    draw_graph(sys.argv[1], None if len(sys.argv) == 2 else sys.argv[2])
