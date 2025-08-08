import json
import math
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plot
from matplotlib.patches import Circle
from matplotlib.widgets import Button
from scipy.spatial import Delaunay

## Exemplary thresholds: [.15, .25, .50, .70, .90]
##                      -> 30   40   50   70   90  100  km/h
##
def quantify_series(data: list[float], thresholds: list[float]) -> list[float]:
    quant = [0] * (len(thresholds) + 1)
    for entry in data:
        for i, th in enumerate(thresholds):
            if entry < th:
                quant[i] += 1
                break
        else:
            quant[-1] += 1
    return quant

def generate_quantities(n_cust=10, n_veh=3, prec=10, load=1000, min_frac=.5, max_frac=.8):
    quantities = [prec] * n_customers
    
    
    
    return quantities

class Graph:
    def __init__(self, n_centroids, n_vertices):
        self.centroids = Graph.generate_centroids(n_centroids)
        self.vertices = Graph.generate_vertices(n_vertices, self.centroids)
        self.shift_overlapping_vertices()
        self.edges, faces = Graph.delaunay_triangulation(self.vertices)
        self.reduce_narrow_triangles(faces, 30)
    
    def generate_centroids(count: int) -> list[ tuple[ float, float ] ]:
        # The centroids should be placed in more or less equal angular distances.
        angle = math.tau / count
        base = np.random.rand() * math.tau
        centroids = [ ]
        if count > 3:
            centroids.append((0, 0))
            count -= 1
        for i in range(count):
            # Generate the phase
            phase_mean = base + i * angle
            phase_sdev = angle / 6
            phase = np.random.normal(phase_mean, phase_sdev, 1)[0]
            # Generate the radius
            RADIUS = 5_000   # in meters
            RADIUS_MEAN, RADIUS_SDEV = RADIUS/2, RADIUS/6
            radius = np.random.normal(RADIUS_MEAN, RADIUS_SDEV, 1)[0]
            radius = max(0, min(radius, RADIUS))
            # Translate to rectangular coordinates
            c = (radius * math.cos(phase), radius * math.sin(phase))
            centroids.append(c)
        return centroids
    
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
    
    def generate_vertices(count: int, centroids: list[ tuple[float, float] ]) -> list[ tuple[float, float] ]:
        vertices = [ ]
        distances = Graph.nearest_neighbor_distances(centroids)
        allocation = Graph.dhondt_method(count, distances)
        for c in range(len(centroids)):
            x_cen, y_cen = centroids[c][0], centroids[c][1]
            d = distances[c]
            n_ver = allocation[c]
            for _ in range(n_ver):
                ph = np.random.rand() * math.tau
                MEAN = d/2
                SDEV = d/6
                r = np.random.normal(MEAN, SDEV, 1)[0]
                v = (x_cen + r * math.cos(ph), y_cen + r * math.sin(ph))
                vertices.append(v)
        return vertices
    
    def shift_overlapping_vertices(self):
        MIN_DIST = 100
        SEMIDIAGONAL = 0.5 * MIN_DIST / math.sqrt(2)
        while True:
            overlaps = [ ]
            for i in range(len(self.vertices)):
                for j in range(i + 1, len(self.vertices)):
                    (x1, y1), (x2, y2) = self.vertices[i], self.vertices[j]
                    if x1 == x2 and y1 == y2:
                        overlaps.append((i, j))
            if not overlaps:
                break
            for i, j in overlaps:
                (x1, y1), (x2, y2) = self.vertices[i], self.vertices[j]
                self.vertices[i] = (x1 - SEMIDIAGONAL, y1 - SEMIDIAGONAL)
                self.vertices[j] = (x2 + SEMIDIAGONAL, y2 + SEMIDIAGONAL)
    
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
        a = np.degrees( np.arccos( (y*y + z*z - x*x) / (2*y*z) ) )
        b = np.degrees( np.arccos( (z*z + x*x - y*y) / (2*z*x) ) )
        c = np.degrees( np.arccos( (x*x + y*y - z*z) / (2*x*y) ) )
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
        avg_dist = [self.average_neighbor_distance(w) for w in range(len(self.vertices))]
        for (u, v) in self.edges:
            congestion_indices.append((avg_dist[u] + avg_dist[v]) / 2)
        lo, hi = min(congestion_indices), max(congestion_indices)
        if hi != lo:
            a = 1 / (hi - lo)
            b = -a * lo
            congestion_indices = [ a * x + b for x in congestion_indices ]
        else:
            congestion_indices = [ 0.5 ] * len(self.edges)
        return congestion_indices
    
    def compute_travel_times(self):
        self.congestion = self.edge_congestions( )
    
    def distribute_orders(self):
        self.quantities = [ 0 ] * len(self.vertices)
        
    def vertex_colors(self):
        avg_dist = [self.average_neighbor_distance(i) for i in range(len(self.vertices))]
        avg_dist_valid = [ x for x in avg_dist if x is not None ]
        lo, hi = min(avg_dist_valid), max(avg_dist_valid)
        
        if hi != lo:
            a = 1 / (hi - lo)
            b = -a * lo
            avg_dist = [ a * x + b if x is not None else None for x in avg_dist ]
            return [
                mcolors.hsv_to_rgb([x/3, 1, 1]) if x is not None else (.75, .75, .75)
                for x in avg_dist
            ]
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
        xc, yc = zip(*self.centroids)
        axes.plot(xc, yc, 'c+')
        # Vertices
        def draw_vertex(i, x, y, edge_color='black'):
            circle = Circle((x, y), 100, color='white', ec=edge_color, linewidth=2)
            axes.add_patch(circle)
            axes.text(x, y, str(i), ha='center', va='center', fontsize=10, color='black', zorder=2)
        colors = self.vertex_colors()
        for i in range(len(self.vertices)):
            draw_vertex(i, self.vertices[i][0], self.vertices[i][1], colors[i])
        # Plot settings
        axes.set_title(f'Graph: {len(self.centroids)} centroids, {len(self.vertices)} vertices')
        axes.set_aspect('equal', adjustable='datalim')
        plot.draw()
    
    def dump(self):
        top = { }
        top['vertices'] = { }
        for i, v in enumerate(self.vertices):
            v_obj = { "x": round(v[0], 3), 
                      "y": round(v[1], 3),
                      "q": self.quantities[i] }
            top['vertices'][str(i)] = v_obj
        top['edges'] = [ ]
        for e in self.edges:
            e_obj = { "u": str(e[0]),
                      "v": str(e[1]) }
            top['edges'].append(e_obj)
        return json.dumps(top, indent=4)

def params(first, last, multipliers):
    list = [ ]
    for c in range(first, last + 1):
        for m in multipliers:
            list.append((c, int(m * c)))
    return list

def main():
    graphs = [ ]
    
    FIRST =  8
    LAST  = 20
    MULTS = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6]
    
    for c, v in params(FIRST, LAST, MULTS):
        graph = Graph(c, v)
        graph.compute_travel_times( )
        graph.distribute_orders( )
        graphs.append(graph)
    
    figure, axes = plot.subplots()
    plot.subplots_adjust(bottom = 0.2)
    
    graphs[0].plot(axes)
    
    class Index:
        def __init__(self, length):
            self.index = 0
            self.length = length
        
        def next(self, event):
            self.index = self.index + 1 if self.index < self.length - 1 else 0
            graphs[self.index].plot(axes)
        def prev(self, event):
            self.index = self.index - 1 if self.index > 0 else self.length - 1
            graphs[self.index].plot(axes)
        def dump(self, event):
            print(graphs[self.index].dump( ))
    
    callback = Index((LAST - FIRST + 1) * len(MULTS))
    axes_dump = plot.axes([0.6, 0.05, 0.1, 0.075])
    axes_prev = plot.axes([0.7, 0.05, 0.1, 0.075])
    axes_next = plot.axes([0.8, 0.05, 0.1, 0.075])
    button_dump = Button(axes_dump, 'Dump')
    button_dump.on_clicked(callback.dump)
    button_prev = Button(axes_prev, 'Prev')
    button_prev.on_clicked(callback.prev)
    button_next = Button(axes_next, 'Next')
    button_next.on_clicked(callback.next)
    
    plot.show()
    
main()

# Do zrobienia:
#  + generowanie czasu przejazdu
#  + generowanie zapotrzebowania klientów
