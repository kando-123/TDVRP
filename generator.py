# NEW

import json
import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plot
import numpy as np
import random
import sys
import os
from abc import ABC, abstractmethod
from matplotlib.patches import Circle
from matplotlib.widgets import Button
from scipy.spatial import Delaunay

class VertexGenerator:
    def generate(n_vertices: int) -> list[ tuple[float, float] ]:
        pass
    
class OrthogonalVertexGenerator(VertexGenerator):
    def __init__(self, orth_radius):
        self.orth_radius = orth_radius
    
    def generate(self, n_vertices):
        vertices = set( )
        while len(vertices) < n_vertices:
            x = random.randint(-self.orth_radius, +self.orth_radius)
            y = random.randint(-self.orth_radius, +self.orth_radius)
            vertices.add((x, y))
        return list(vertices)

class CentroidVertexGenerator(VertexGenerator):
    def generate_centroids(count, max_radius) -> list[ tuple[float, float] ]:
        centroids = set( )
        if count > 3:
            centroids.add((0, 0))
            count -= 1
        # The centroids should be placed in more or less equal angular distances.
        angle = math.tau / count
        base = random.random() * math.tau
        for i in range(count):
            # Generate the phase
            phase_mean = base + i * angle
            phase_sdev = angle / 6
            phase = random.gauss(phase_mean, phase_sdev)
            # Generate the radius
            radius_mean, radius_sdev = max_radius / 2, max_radius / 4
            radius = random.gauss(radius_mean, radius_sdev)
            radius = max(0, min(radius, max_radius))
            # Translate to rectangular coordinates
            c = (radius * math.cos(phase), radius * math.sin(phase))
            centroids.add(c)
        return list(centroids)
        
    def __init__(self, n_centroids, max_radius):
        self.centroids = CentroidVertexGenerator.generate_centroids(n_centroids, max_radius)
    
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
    
    def dhondt_method(seats: int, votes: list[float]) -> list[int]:
        n_parties = len(votes)
        if seats < n_parties:
            raise ValueError('Unable to allocate at least one seat for each party!')
        allocation = [ 1 ] * n_parties
        seats -= n_parties
        for _ in range(seats):
            scores = [ votes[i] / allocation[i] for i in range(n_parties) ]
            index = scores.index(max(scores))
            allocation[index] += 1
        return allocation
    
    def shift_overlaps(vertices):
        epsilon = 1
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
                vertices[i] = (x1 - epsilon, y1 - epsilon)
                vertices[j] = (x2 + epsilon, y2 + epsilon)
    
    def generate(self, n_vertices):
        vertices = [ ]
        distances = CentroidVertexGenerator.nearest_neighbor_distances(self.centroids)
        allocation = CentroidVertexGenerator.dhondt_method(n_vertices, distances)
        for i, (x, y) in enumerate(self.centroids):
            d = distances[i]
            n_ver = allocation[i]
            ver = set( )
            while len(ver) < n_ver:
                ph = random.random() * math.tau
                mean = d/4
                sdev = d/12
                r = random.gauss(mean, sdev)
                v = (round(x + r * math.cos(ph)), round(y + r * math.sin(ph)))
                ver.add(v)
            vertices.extend(ver)
        CentroidVertexGenerator.shift_overlaps(vertices)
        return vertices

def make_generator(generation_config) -> VertexGenerator:
    type = generation_config["type"]
    generator = None
    if type == 'centroid':
        n_centroids = generation_config["n_centroids"]
        max_radius = generation_config["max_radius"]
        generator = CentroidVertexGenerator(n_centroids, max_radius)
    elif type == 'orthogonal':
        radius = generation_config["radius"]
        generator = OrthogonalVertexGenerator(radius)
    else:
        raise Exception('Unknown type of genertation')
    return generator

class Graph:
    def __init__(self, depot=0):
        self.vertices   = list( )
        self.orders     = list( )   # i-th element is the order of the i-th vertex
        self.edges      = list( )
        self.weights    = list( )   # i-th element is the list of weights of the i-th edge
        self.depot      = depot
        self.n_vehicles = None
        self.max_load   = None
        self.interval   = None
    
    def generate_body(self, n_vertices, generator, min_angle_deg):
        self.vertices = generator.generate(n_vertices)
        self.edges = Graph.connect(self.vertices, min_angle_deg)
    
    def connect(vertices: list[ tuple[float, float] ], min_angle_deg: float) -> list[ tuple[int, int] ]:
        points = np.array(vertices)
        tri = Delaunay(points)
        simplices = list(map(tuple, tri.simplices))
        good_edges, bad_edges = set( ), set( )
        for simplex in simplices:
            good, bad = Graph.get_edges(simplex, vertices, min_angle_deg)
            good_edges = good_edges | good
            bad_edges  = bad_edges  | bad
        return list(good_edges - bad_edges)
    
    def get_edges(simplex: tuple[int, int, int], vertices: list[tuple[float, float]], min_angle_deg: float) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        sides = [ ]
        for i in range(3):
            u, v = vertices[simplex[(i + 1) % 3]], vertices[simplex[(i + 2) % 3]]
            x, y = u[0] - v[0], u[1] - v[1]
            sides.append(math.hypot(x, y))
        angles_deg = [ ]
        for i in range(3):
            a = sides[i]
            b = sides[(i + 1) % 3]
            c = sides[(i + 2) % 3]
            cosine = (b*b + c*c - a*a) / (2*b*c)
            angles_deg.append(math.degrees(math.acos(cosine)))
        good, bad = set( ), set( )
        if min(angles_deg) < min_angle_deg:
            for i in range(3):
                edge = (simplex[(i + 1) % 3], simplex[(i + 2) % 3])
                if edge[0] > edge[1]:
                    edge = (edge[1], edge[0])
                if sides[i] > sides[(i + 1) % 3] and sides[i] >= sides[(i + 2) % 3]:
                    # The longest side is sides[i]
                    bad.add(edge)
                else:
                    # It is not the longest side
                    good.add(edge)
        else:
            for i in range(3):
                edge = (simplex[(i + 1) % 3], simplex[(i + 2) % 3])
                if edge[0] > edge[1]:
                    edge = (edge[1], edge[0])
                good.add(edge)
        return good, bad
    
    def distribute_orders(self, total_load, precision, n_vehicles, max_load):
        self.n_vehicles = n_vehicles
        self.max_load   = max_load
        self.orders.clear( )
        quantities = Graph.generate_orders(len(self.vertices) - 1, total_load, precision)
        # while not Graph.validate(quantities, n_vehicles, max_load):
        #     quantities = Graph.generate_orders(len(self.vertices) - 1, total_load, precision)
        j = 0
        for i in range(len(self.vertices)):
            if i == self.depot:
                self.orders.append(0)
            else:
                self.orders.append(quantities[j])
                j += 1
    
    def generate_orders(n_customers, max_total_load, precision) -> list[float]:
        orders = [precision] * n_customers
        total_load = precision * n_customers
        while total_load + precision < max_total_load:
            index = random.randrange(n_customers)
            orders[index] += precision
            total_load += precision
        return orders
    
    def compute_weights(self, rush_coeff: list[float], thresholds: list[float], limits_mps: list[float], interval=None):
        self.interval = interval if interval is not None else 86_400 / len(rush_coeff)
        congestion = self.calculate_congestion( )
        travel_times = [ None ] * len(self.edges)
        for k, e in enumerate(self.edges):
            limit = limits_mps[ Graph.interval_index(congestion[k], thresholds) ]
            (i, j) = e
            (x1, y1), (x2, y2) = self.vertices[i], self.vertices[j]
            length = math.hypot(x2 - x1, y2 - y1)
            travel_times[k] = [ round(length / Graph.speed(limit, congestion[k], r)) for r in rush_coeff ]   # t = s/v
        self.weights = travel_times
    
    def calculate_congestion(self):
        congestion_indices = [ ]
        avg_dist = [ self.average_neighbor_distance(w) for w in range(len(self.vertices)) ]
        for (u, v) in self.edges:
            congestion_indices.append((avg_dist[u] + avg_dist[v]) / 2)
        lo, hi = min(congestion_indices), max(congestion_indices)
        if hi != lo:
            # Scale to interval [0, 1] (both included)
            a = 1 / (hi - lo)
            b = -a * lo
            for i in range(len(congestion_indices)):
                congestion_indices[i] = a * congestion_indices[i] + b
        else:
            congestion_indices = [ 0.5 ] * len(self.edges)
        return congestion_indices
    
    def average_neighbor_distance(self, index):
        neighbors = set( )
        for (i, j) in self.edges:
            if i == index:
                neighbors.add(self.vertices[i])
            elif j == index:
                neighbors.add(self.vertices[j])
        return Graph.average_distance(self.vertices[index], neighbors)
    
    def average_distance(vertex, neighbors) -> float:
        if len(neighbors) > 0:
            (x0, y0) = vertex
            sum = 0
            for (x, y) in neighbors:
                sum += math.hypot(x - x0, y - y0)
            return sum / len(neighbors)
        else:
            return None
    
    def interval_index(value: float, thresholds: list[float]) -> int:
        for i, th in enumerate(thresholds):
            if value < th:
                return i
        return -1   # Index of the last in negative indexation;
                    # len(thresholds) might not be used as a valid index.
    
    BASE = 1/math.sqrt(0.2)
    
    def speed(speed_limit, congestion, rush):
        return speed_limit * Graph.BASE ** -(congestion + rush)
    
    def dump(self):
        top = { }
        top["vertices"] = { }
        for i, v in enumerate(self.vertices):
            v_obj = { "x": round(v[0], 3), 
                      "y": round(v[1], 3),
                      "q": self.orders[i] }
            top["vertices"][str(i)] = v_obj
        top["edges"] = [ ]
        for i, e in enumerate(self.edges):
            e_obj1 = {
                "u": str(e[0]),
                "v": str(e[1]),
                "w": self.weights[i]
            }
            e_obj2 = {
                "u": str(e[1]),
                "v": str(e[0]),
                "w": self.weights[i]
            }
            top["edges"].append(e_obj1)
            top["edges"].append(e_obj2)
        top["fleet"] = {
            "n_vehicles": self.n_vehicles,
            "max_load":   self.max_load
        }
        top["interval"] = self.interval
        return json.dumps(top, indent=4)
    
    def boundaries(lol: list[list[float]]) -> tuple[float, float]:
        minimum = float('inf')
        maximum = float('-inf')
        for l in lol:
            for el in l:
                if el < minimum:
                    minimum = el
                elif el > maximum:
                    maximum = el
        return minimum, maximum
    
    def linear_transform_coefficients(x1, x2, y1, y2):
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        return a, b
    
    def plot(self, axes):
        axes.clear( )
        # Edges
        minimum, maximum = Graph.boundaries(self.weights)
        a, b = Graph.linear_transform_coefficients(minimum, maximum, 0, 1)
        # Weight[min, max] => [0, 1]
        def transform(w):
            return a * w + b
        colors = [ mcolors.hsv_to_rgb([transform(w[0]) / 3, 1, 1]) for w in self.weights ]
        for k, e in enumerate(self.edges):
            (i, j) = e
            (x1, y1) = self.vertices[i]
            (x2, y2) = self.vertices[j]
            axes.plot([x1, x2], [y1, y2], '-', color=colors[k], zorder=1)
        # Vertices
        def draw_vertex(coords, is_depot=False):
            color = 'blue' if is_depot else 'white'
            circle = Circle(coords, 200, color=color, ec='black', linewidth=1)
            axes.add_patch(circle)
        for i, v in enumerate(self.vertices):
            draw_vertex(v, i == self.depot)
        # Plot settings
        axes.set_title(f'Graph: {len(self.vertices)} vertices')
        axes.set_aspect('equal', adjustable='datalim')
        plot.draw( )

def main(config):
    graphs = [ ]
    
    n_vertices_list, n_examples_list = config["n_vertices"], config["n_examples"]
    body = config["body"]
    orders = config["orders"]
    weights = config["weights"]
    
    for n_vertices, n_examples in zip(n_vertices_list, n_examples_list):
        for _ in range(n_examples):
            graph = Graph( )
            graph.generate_body(n_vertices, make_generator(body), config["min_angle_deg"])
            graph.distribute_orders(orders["total"], orders["precision"], orders["n_vehicles"], orders["max_load"])
            graph.compute_weights(weights["rush_coeff"], weights["thresholds"], weights["limits_mps"], config["interval_s"])
            graphs.append(graph)

    figure, axes = plot.subplots( )
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
            core = sys.argv[2] if len(sys.argv) > 2 else 'graph'
            name = core
            i = 0
            while os.path.isfile(f'{name}.json'):
                i += 1
                name = f'{core}-{i}'
            with open(f'{name}.json', 'w') as file:
                file.write(graphs[self.index].dump( ))
                print(f'Saved as: "{name}.json"')
    
    callback = Index(len(graphs))
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

if len(sys.argv) < 2:
    print('Usage: python generator.py configu.json')
else:
    with open(sys.argv[1]) as config_file:
        config = json.load(config_file)
    main(config)