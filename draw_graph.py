import sys
import json
import matplotlib.pyplot as plot
import os
from matplotlib.patches import Circle
from math import hypot, sin, cos
from math import atan2 as phase

delta = 0.2

def draw_vertex(axes, x, y, index, quantity, radius, is_depot=False):
    circle = Circle((x, y), radius, color='lightcoral' if is_depot else 'white', ec='black', linewidth=2, zorder=2)
    axes.add_patch(circle)
    if quantity > 0:
        axes.plot([x - radius, x + radius], [y, y], color='black', lw=1, zorder=3)
        axes.text(x, y + radius * 0.4, str(index), ha='center', va='center', fontsize=8, color='black', zorder=4)
        axes.text(x, y - radius * 0.6, str(quantity), ha='center', va='center', fontsize=8, color='red', zorder=4)
    else:
        axes.text(x, y, str(index), ha='center', va='center', fontsize=10, zorder=4, color='black')

def draw_edge(axes, x1, y1, x2, y2, radius, color='gray'):
    (x, y) = (x2 - x1, y2 - y1)
    ph = phase(y, x)
    (dx, dy) = (delta * -sin(ph), delta * cos(ph))
    r = hypot(x, y)
    (Dx, Dy) = (x * radius / r, y * radius / r)
    axes.annotate('', (x2 - Dx + dx, y2 - Dy + dy), (x1 + Dx + dx, y1 + Dy + dy),
                  arrowprops=dict(arrowstyle='->', color=color, lw=2),
                  zorder=1)

def interpolate_color(value, min, max):
    if max == min:
        return (0.5, 0.5, 0.5)   # gray if all equal
    ratio = (value - min) / (max - min)
    return (ratio, 1 - ratio, 0)   # linear interpolation from green to red

def get_radius(axes, n_vertices, scaling_factor=0.03):
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()
    # Calculate the smaller dimension of your plotting area
    plot_width = xlim[1] - xlim[0]
    plot_height = ylim[1] - ylim[0]
    min_dim = min(plot_width, plot_height)
    # Make radius proportional; scaling_factor adjusts size, tweak as needed
    radius = min_dim # * scaling_factor / (n_vertices ** 0.5)
    return radius

def draw_graph(in_file, out_dir):
    with open(in_file) as f:
        graph = json.load(f)

    vertices = graph['vertices']
    edges = graph['edges']
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

        r = get_radius(axes, len(vertices))
        
        for e in edges:
            (u, v) = (e['u'], e['v'])
            (x1, y1) = (vertices[u]['x'], vertices[u]['y'])
            (x2, y2) = (vertices[v]['x'], vertices[v]['y'])
            weight = e['w'][index] if 'w' in e else None
            color = interpolate_color(weight, w_min, w_max) if weight is not None else 'gray'
            draw_edge(axes, x1, y1, x2, y2, r, color)
        
        for i, v in vertices.items( ):
            (x, y) = (v['x'], v['y'])
            q = v.get('q', 0)
            draw_vertex(axes, x, y, i, q, r, is_depot=(i == depot))

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
