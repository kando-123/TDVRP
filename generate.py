import json
import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plot
import numpy as np
import sys
import os
from abc import ABC, abstractmethod
from matplotlib.patches import Circle
from matplotlib.widgets import Button
from scipy.spatial import Delaunay

class Generator(ABC):
    @abstractmethod
    def generate(n_vertices: int) -> list[tuple[float, float]]:
        pass

class OrthogonalGenerator(Generator):
    def __init__(self, orth_radius):
        self.orth_radius = orth_radius
    
    def generate_vertices(n_vertices):
        vertices = set( )
        while len(vertices) < n_vertices:
            x = np.random.randint(-self.orth_radius, self.orth_radius + 1)
            y = np.random.randint(-self.orth_radius, self.orth_radius + 1)
            vertices.add((x, y))
        return list(vertices)

class CentroidGenerator(Generator):
    def generate_centroids(self, count: int, max_radius=10_000) -> list[ tuple[ float, float ] ]:
        # The centroids should be placed in more or less equal angular distances.
        angle = math.tau / count
        base = np.random.rand() * math.tau
        centroids = set( )
        if count > 3:
            centroids.add((0, 0))
            count -= 1
        for i in range(count):
            # Generate the phase
            phase_mean = base + i * angle
            phase_sdev = angle / 6
            phase = np.random.normal(phase_mean, phase_sdev, 1)[0]
            # Generate the radius
            radius_mean, radius_sdev = max_radius/2, max_radius/6
            radius = np.random.normal(radius_mean, radius_sdev, 1)[0]
            radius = max(0, min(radius, max_radius))
            # Translate to rectangular coordinates
            c = (radius * math.cos(phase), radius * math.sin(phase))
            centroids.add(c)
        return list(centroids)
        
    def __init__(self, n_centroids, max_radius=10_000):
        self.centroids = self.generate_centroids(n_centroids, max_radius)
    
    def nearest_neighbor_distances(points: list[ tuple[float, float] ]) -> list[float]:
        
        def distance_square(p, q):
            return (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1])
        
        distances = [ ]
        for i, p in enumerate(points):
            min_dist = float('inf')
            for j, q in enumerate(points):
                if i == j:
                    continue
                dist = distance_square(p, q)
                if dist < min_dist:
                    min_dist = dist
            distances.append(min_dist)
        for i in range(len(distances)):
            distances[i] = math.sqrt(distances[i])
        return distances
    
    # Variant that assigns at least one seat to each party
    def dhondt_method(seats: int, votes: list[float]) -> list[int]:
        n_parties = len(votes)
        if seats < n_parties:
            raise ValueError('Unable to allocate at least one seat for each party!')
        allocation = [1] * n_parties
        seats -= n_parties
        for _ in range(seats):
            scores = [ votes[i] / allocation[i] for i in range(n_parties) ]
            index = scores.index(max(scores))
            allocation[index] += 1
        return allocation
    
    def shift_overlaps(vertices):
        min_dist = 1
        semidiagonal = 0.5 * min_dist / math.sqrt(2)
        while True:
            overlaps = [ ]
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    (x1, y1), (x2, y2) = vertices[i], vertices[j]
                    if x1 == x2 and y1 == y2:
                        overlaps.append((i, j))
            if not overlaps:
                break
            for i, j in overlaps:
                (x1, y1), (x2, y2) = vertices[i], vertices[j]
                vertices[i] = (x1 - semidiagonal, y1 - semidiagonal)
                vertices[j] = (x2 + semidiagonal, y2 + semidiagonal)
    
    def generate(self, n_vertices):
        vertices = [ ]
        distances = CentroidGenerator.nearest_neighbor_distances(self.centroids)
        allocation = CentroidGenerator.dhondt_method(n_vertices, distances)
        for i, c in enumerate(self.centroids):
            x_cen, y_cen = c[0], c[1]
            d = distances[i]
            n_ver = allocation[i]
            ver = set( )
            while len(ver) < n_ver:
                ph = np.random.rand() * math.tau
                mean = d/4
                sdev = d/12
                r = np.random.normal(mean, sdev, 1)[0]
                v = (round(x_cen + r * math.cos(ph)), round(y_cen + r * math.sin(ph)))
                ver.add(v)
            vertices.extend(ver)
        CentroidGenerator.shift_overlaps(vertices)
        return vertices

def generate_quantities(n_cust=10, n_veh=3, prec=10, load=1000, lo_frac=.5, hi_frac=.8):
    quantities = [1] * n_cust
    max_units = n_veh * load / prec - n_cust
    for _ in range(round(.5 * (lo_frac + hi_frac) * max_units)):
        quantities[np.random.randint(n_cust)] += 1
    for i, val in enumerate(quantities):
        quantities[i] = prec * val
    return quantities

class Graph:
    def __init__(self, n_vertices, **kwargs):
        
        generation = kwargs.get('generation', 'centroid')
        if generation == 'centroid':
            self.generator = CentroidGenerator(kwargs['n_centroids'])
        elif generation == 'orthogonal':
            self.generator = OrthogonalGenerator(kwargs['orth_radius'])
        else:
            raise ValueError('Unknown type of generation')
        
        self.vertices = self.generator.generate(n_vertices)
        
        self.edges, faces = Graph.delaunay_triangulation(self.vertices)
        self.reduce_narrow_triangles(faces, 30)
        
        self.depot = kwargs.get('depot', 0)
    
    def delaunay_triangulation(vertices: list[ tuple[float, float] ]):
        points = np.array(vertices)
        tri = Delaunay(points)
        edges = list( )
        for simplex in tri.simplices:
            for i in range(3):
                u, v = simplex[i], simplex[(i + 1) % 3]
                e = (u, v) if u < v else (v, u)
                edges.append(e)
        faces = [ list(simplex) for simplex in tri.simplices ]
        return edges, faces
    
    def handle_simplex(self, tri):
        i, j, k = tri[0], tri[1], tri[2]
        u, v, w = self.vertices[i], self.vertices[j], self.vertices[k]
        x = math.hypot(v[0] - w[0], v[1] - w[1])
        y = math.hypot(w[0] - u[0], w[1] - u[1])
        z = math.hypot(u[0] - v[0], u[1] - v[1])
        a = np.degrees(np.arccos((y * y + z * z - x * x) / (2 * y * z)))
        b = np.degrees(np.arccos((z * z + x * x - y * y) / (2 * z * x)))
        c = np.degrees(np.arccos((x * x + y * y - z * z) / (2 * x * y)))
        edges = [ ]
        if min(a, b, c) < 20:
            if x > y and x >= z:
                edges.append((j, k))
                edges.append((k, j))
            elif y > z and y >= x:
                edges.append((k, i))
                edges.append((i, k))
            elif z > x and z >= y:
                edges.append((i, j))
                edges.append((j, i))
        return edges
    
    def reduce_narrow_triangles(self, faces, threshold_deg=20):
        removables = [ ]
        for face in faces:
            removables.extend(self.handle_simplex(face))
        self.edges = list(set(self.edges) - set(removables))
    
    def average_neighbor_distance(self, vertex):
        neighbors = set( )
        for e in self.edges:
            if e[0] == vertex:
                neighbors.add(e[1])
            elif e[1] == vertex:
                neighbors.add(e[0])
        (x0, y0) = self.vertices[vertex]
        length = 0
        for n in neighbors:
            (x, y) = self.vertices[n]
            length += math.hypot(x - x0, y - y0)
        return length / len(neighbors) if len(neighbors) > 0 else None
    
    def edge_congestions(self):
        congestion_indices = [ ]
        avg_dist = [ self.average_neighbor_distance(w) for w in range(len(self.vertices)) ]
        for (u, v) in self.edges:
            congestion_indices.append((avg_dist[u] + avg_dist[v]) / 2)
        lo, hi = min(congestion_indices), max(congestion_indices)
        if hi != lo:
            a = 1 / (hi - lo)
            b = -a * lo
            for i in range(len(congestion_indices)):
                congestion_indices[i] = a * congestion_indices[i] + b
        else:
            congestion_indices = [ 0.5 ] * len(self.edges)
        return congestion_indices
    
    def slot_index(value, thresholds: list[float]):
        limit = None
        for i, th in enumerate(thresholds):
            if value < th:
                limit = i
                break
        else:
            limit = -1
        return limit
    
    BASE = 1/math.sqrt(0.2)
    
    def speed(speed_limit, congestion, rush):
        return speed_limit * Graph.BASE ** -(congestion + rush)
    
    def compute_travel_times(self, rush: list[float], thresholds: list[float], limits: list[float]):
        self.congestion = self.edge_congestions( )
        travel_times = [ None ] * len(self.edges)
        for i, e in enumerate(self.edges):
            limit = limits[ Graph.slot_index(self.congestion[i], thresholds) ]
            (u, v) = e
            (x1, y1), (x2, y2) = self.vertices[u], self.vertices[v]
            length = math.hypot(x1 - x2, y1 - y2)
            # t = s/v
            travel_times[i] = [ round(length / Graph.speed(limit, self.congestion[i], r)) for r in rush ]
        self.weights = travel_times
    
    def distribute_orders(self, n_ord):
        quantities = generate_quantities(n_ord)
        indexes = list(range(1, len(self.vertices)))
        self.orders = { }
        for i, q in enumerate(quantities):
            index = np.random.randint(len(indexes))
            indexes[index], indexes[-1] = indexes[-1], indexes[index]
            self.orders[indexes.pop()] = quantities[i]
        
    def vertex_colors(self):
        avg_dist = [self.average_neighbor_distance(i) for i in range(len(self.vertices))]
        avg_dist_valid = [ x for x in avg_dist if x is not None ]
        lo, hi = min(avg_dist_valid), max(avg_dist_valid)
        
        if hi != lo:
            a = 1 / (hi - lo)
            b = -a * lo
            for i in range(len(avg_dist)):
                avg_dist[i] = a * avg_dist[i] + b if avg_dist[i] is not None else None
            return [ mcolors.hsv_to_rgb([x/3, 1, 1]) if x is not None else (.75, .75, .75) for x in avg_dist ]
        else:
            return [ (0, 0, 0) ] * len(self.vertices)
    
    def plot(self, axes):
        axes.clear()
        # Edges
        colors = [ mcolors.hsv_to_rgb( [c/3, 1, 1] ) for c in self.congestion ]
        for e in range(len(self.edges)):
            (u, v) = self.edges[e]
            (x1, y1) = self.vertices[u]
            (x2, y2) = self.vertices[v]
            axes.plot([x1, x2], [y1, y2], '-', color=colors[e], zorder=1)
        # Centroids
        xc, yc = zip(*self.generator.centroids)
        axes.plot(xc, yc, 'c+')
        # Vertices
        def draw_vertex(i, edge_color='black'):
            (x, y) = self.vertices[i]
            circle = Circle((x, y), 250,
                color='magenta' if i == int(self.depot) else 'blue' if i in self.orders else 'white',
                ec=edge_color, linewidth=1)
            axes.add_patch(circle)
            # axes.text(x, y, str(i), ha='center', va='center', fontsize=10, color='black' if i not in self.orders else 'blue', zorder=2)
        colors = self.vertex_colors()
        for i in range(len(self.vertices)):
            draw_vertex(i, colors[i])
        # Plot settings
        axes.set_title(f'Graph: {len(self.generator.centroids)} centroids, {len(self.vertices)} vertices')
        # axes.set_title(f'Graph: {len(self.vertices)} vertices')
        axes.set_aspect('equal', adjustable='datalim')
        plot.draw( )
    
    def dump(self):
        top = { }
        top['vertices'] = { }
        for i, v in enumerate(self.vertices):
            v_obj = { "x": round(v[0], 3), 
                      "y": round(v[1], 3) }
            order = self.orders.get(i)
            if order is not None:
                v_obj["q"] = order
            top['vertices'][str(i)] = v_obj
        top['edges'] = [ ]
        for i, e in enumerate(self.edges):
            e_obj1 = { "u": str(e[0]),
                       "v": str(e[1]),
                       "w": self.weights[i] }
            e_obj2 = { "u": str(e[1]),
                       "v": str(e[0]),
                       "w": self.weights[i] }
            top['edges'].append(e_obj1)
            top['edges'].append(e_obj2)
            
        return json.dumps(top, indent=4)

def main():
    graphs = [ ]
    
    def params(first, last, multipliers):
        list = [ ]
        for c in range(first, last + 1):
            for m in multipliers:
                list.append((c, int(m * c)))
        return list
    
    FIRST = 10
    LAST  = 20
    MULTS = [3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]
    
    RUSH = [1.0, 0.2, 0.4, 1.0, 0.6, 0.2, 0.0, 0.0]
    THLD = [   .15, .25, .50, .70, .90    ]
    LIMS = [ 30,  40,  50,  70,  90,  100 ]   # km/h
    
    for i, lim in enumerate(LIMS):
        LIMS[i] = lim / 3.6   # m/s

    for c, v in params(FIRST, LAST, MULTS):
        graph = Graph(v, n_centroids=c)
        graph.distribute_orders(v // 3)
        graph.compute_travel_times(RUSH, THLD, LIMS)
        graphs.append(graph)
    
    figure, axes = plot.subplots()
    plot.subplots_adjust(bottom = 0.2)
    
    graphs[0].plot(axes)
    
    class Index:
        def __init__(self, length):
            self.index = 0
            self.end = length - 1
        
        def next(self, event):
            self.index = self.index + 1 if self.index < self.end else 0
            graphs[self.index].plot(axes)
        
        def prev(self, event):
            self.index = self.index - 1 if self.index > 0 else self.end
            graphs[self.index].plot(axes)
        
        def dump(self, event):
            print(graphs[self.index].dump( ))
        
        def save(self, event):
            core = sys.argv[1] if len(sys.argv) > 1 else 'out'
            name = core
            i = 0
            while os.path.isfile(f'{name}.json'):
                i += 1
                name = f'{core} ({i})'
            with open(f'{name}.json', 'w') as file:
                file.write(graphs[self.index].dump( ))
                print(f'Saved as: "{name}.json"')
    
    callback = Index((LAST - FIRST + 1) * len(MULTS))
    axes_save = plot.axes([0.1, 0.05, 0.1, 0.075])
    axes_dump = plot.axes([0.2, 0.05, 0.1, 0.075])
    axes_prev = plot.axes([0.7, 0.05, 0.1, 0.075])
    axes_next = plot.axes([0.8, 0.05, 0.1, 0.075])
    button_save = Button(axes_save, 'Save')
    button_dump = Button(axes_dump, 'Dump')
    button_prev = Button(axes_prev, 'Prev')
    button_next = Button(axes_next, 'Next')
    button_save.on_clicked(callback.save)
    button_dump.on_clicked(callback.dump)
    button_prev.on_clicked(callback.prev)
    button_next.on_clicked(callback.next)
    
    plot.show()

main()