import json
import math
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.patches import Circle
from matplotlib.widgets import Button
from scipy.spatial import Delaunay

class Graph:
    def __init__(self, n_centroids, n_vertices):
        self.centroids = Graph.generate_centroids(n_centroids)
        self.vertices = Graph.generate_vertices(n_vertices, self.centroids)
        self.edges, faces = Graph.delaunay_triangulation(self.vertices)
        self.reduce_narrow_triangles(faces)
    
    def generate_centroids(count: int) -> list[ tuple[ float, float ] ]:
        # The centroids should be placed in more or less equal angular distances.
        angle = math.tau / count
        base = np.random.rand() * math.tau
        centroids = [ ]
        if count > 3:
            centroids.append( (0, 0) )
            count -= 1
        for i in range(count):
            # Generate the phase
            phase_mean = base + i * angle
            phase_sdev = angle / 6
            phase = np.random.normal(phase_mean, phase_sdev, 1)[0]
            # Generate the radius
            RADIUS_MEAN = 10/2
            RADIUS_SDEV = 10/6
            radius = np.random.normal(RADIUS_MEAN, RADIUS_SDEV, 1)[0]
            radius = max(radius,  0)
            radius = min(radius, 10)
            # Translate to rectangular coordinates
            x, y = radius * math.cos(phase), radius * math.sin(phase)
            centroids.append( (x, y) )
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
            # Handle a single centroid in this loop...
            x_cen, y_cen = centroids[c][0], centroids[c][1]
            d = distances[c]
            n_ver = allocation[c]
            for _ in range(n_ver):
                # ...and in this one, handle a single vertex
                ph = np.random.rand() * math.tau
                MEAN = 0.5 * d
                SDEV = 0.3 * d
                r = np.random.normal(MEAN, SDEV, 1)[0]
                v = (x_cen + r * math.cos(ph), y_cen + r * math.sin(ph))
                vertices.append(v)
        return vertices
    
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
    
    def plot(self, axes):
        axes.clear()
        
        # Edges
        for e in self.edges:
            (u, v) = e
            (x1, y1) = self.vertices[u]
            (x2, y2) = self.vertices[v]
            axes.plot([x1, x2], [y1, y2], 'k-', zorder=1)
        
        # Centroids
        xc, yc = zip(*self.centroids)
        axes.plot(xc, yc, 'r+')
        
        # Vertices
        def draw_vertex(i, x, y):
            circle = Circle((x, y), 0.3, color='white', ec='blue', linewidth=2)
            axes.add_patch(circle)
            axes.text(x, y, str(i), ha='center', va='center', fontsize=10, color='blue', zorder=2)
        
        for i in range(len(self.vertices)):
            draw_vertex(i, self.vertices[i][0], self.vertices[i][1])
        
        # Plot settings
        axes.set_title(f'Graph: {len(self.centroids)} centroids, '
            + f'{len(self.vertices)} vertices')
        axes.set_aspect('equal', adjustable='datalim')
        plot.draw()
    
    def handle_simplex(self, tri):
        i, j, k = tri[0], tri[1], tri[2]
        # print(f'face {i}-{j}-{k}', end=': ')
        u, v, w = self.vertices[i], self.vertices[j], self.vertices[k]
        # print(f'({u[0]:.2f}, {u[1]:.2f})-({v[0]:.2f}, {v[1]:.2f})-({w[0]:.2f}, {w[1]:.2f});')
        x = math.hypot(v[0] - w[0], v[1] - w[1])
        y = math.hypot(w[0] - u[0], w[1] - u[1])
        z = math.hypot(u[0] - v[0], u[1] - v[1])
        # print(f'\tx = |{j}:{k}| = {x:.2f}')
        # print(f'\ty = |{k}:{i}| = {y:.2f}')
        # print(f'\tz = |{i}:{j}| = {z:.2f}')
        a = np.degrees( np.arccos( (y*y + z*z - x*x) / (2*y*z) ) )
        b = np.degrees( np.arccos( (z*z + x*x - y*y) / (2*z*x) ) )
        c = np.degrees( np.arccos( (x*x + y*y - z*z) / (2*x*y) ) )
        # print(f'\ta = ang {i} = {a:.2f} deg')
        # print(f'\tb = ang {j} = {b:.2f} deg')
        # print(f'\tc = ang {k} = {c:.2f} deg')
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

def params(first, last, multipliers):
    list = [ ]
    for c in range(first, last + 1):
        for m in multipliers:
            list.append((c, int(m*c)))
    return list

def main():
    graphs = [ ]
    
    FIRST =  5
    LAST  = 20
    MULTS = [1.6, 2.0, 2.4, 2.8, 3.2]
    
    for c, v in params(FIRST, LAST, MULTS):
        graph = Graph(c, v)
        graphs.append(graph)
    
    figure, axes = plot.subplots()
    plot.subplots_adjust(bottom = 0.2)

    graphs[0].plot(axes)

    class Index:
        def __init__(self, length):
            self.index = 0
            self.length = length
        
        def next(self, event):
            if self.index < self.length - 1:
                self.index += 1
                graphs[self.index].plot(axes)
        def prev(self, event):
            if self.index > 0:
                self.index -= 1
                graphs[self.index].plot(axes)

    callback = Index((LAST - FIRST + 1) * len(MULTS))
    axes_prev = plot.axes([0.7, 0.05, 0.1, 0.075])
    axes_next = plot.axes([0.8, 0.05, 0.1, 0.075])
    button_next = Button(axes_next, 'Next')
    button_next.on_clicked(callback.next)
    button_prev = Button(axes_prev, 'Prev')
    button_prev.on_clicked(callback.prev)

    plot.show()

main()

# Do zrobienia:
#  + rozpoznawanie "śródmiejskich" obszarów
#  + generowanie prędkości i czasu przejazdu
#  + wypisywanie do JSON-a
