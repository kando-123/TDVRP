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
                e = sorted([u, v])
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
        axes.set_title(f'Graph for {len(self.centroids)} centroids')
        axes.set_aspect('equal', adjustable='datalim')
        plot.draw()
    
    # a, b, c - sides of the triangle
    def min_angle_deg(a, b, c):
        α = np.arccos( (b*b + c*c - a*a) / (2*b*c) )
        β = np.arccos( (c*c + a*a - b*b) / (2*c*a) )
        γ = np.pi - (α + β)
        α = np.degrees(α)
        β = np.degrees(β)
        γ = np.degrees(γ)
        return min(α, β, γ)
    
    # triangle: list of three vertices (not their indexes)
    def is_narrow(triangle, threshold_deg):
        A, B, C = [triangle[i] for i in range(3)]
        a = math.hypot(B[0] - C[0], B[1] - C[1])
        b = math.hypot(C[0] - A[0], C[1] - A[1])
        c = math.hypot(A[0] - B[0], A[1] - B[1])
        return Graph.min_angle_deg(a, b, c) < threshold_deg
    
    def longest_side(triangle):
        sides = [ math.hypot(triangle[(i+1)%3][0] - triangle[(i+2)%3][0],
                             triangle[(i+1)%3][1] - triangle[(i+2)%3][1])
                             for i in range(3) ]
        for i in range(3):
            if sides[i] > sides[(i+1)%3] and sides[i] >= sides[(i+2)%3]:
                return [(i+1)%3, (i+2)%3]
        return [0, 1]   # If all are equal
    
    def reduce_narrow_triangles(self, faces, threshold_deg=20):
        for face in faces:
            triangle = [self.vertices[i] for i in face]
            if Graph.is_narrow(triangle, threshold_deg):
                print(f'triangle {face[0]}-{face[1]}-{face[2]} is narrow')
                (m, n) = Graph.longest_side(triangle)   # m, n in 0:2
                e = sorted([face[m], face[n]])
                self.edges.remove(e)

def multiple_range(begin, end, samples):
    list = [ ]
    for i in range(begin, end):
        list += [i] * samples
    return list

def main():
    graphs = [ ]
    
    BEGIN   =  2
    END     = 13
    SAMPLES =  3
    VERTICES_PER_CENTROID = 2.5
    
    for n in multiple_range(BEGIN, END, SAMPLES):
        graph = Graph(n, int(VERTICES_PER_CENTROID * n))
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

    callback = Index((END - BEGIN) * SAMPLES)
    axes_prev = plot.axes([0.7, 0.05, 0.1, 0.075])
    axes_next = plot.axes([0.8, 0.05, 0.1, 0.075])
    button_next = Button(axes_next, 'Next')
    button_next.on_clicked(callback.next)
    button_prev = Button(axes_prev, 'Prev')
    button_prev.on_clicked(callback.prev)

    plot.show()

main()

# Do zrobienia:
#  + łączenie wierzchołków krawędziami
#  + rozpoznawanie "śródmiejskich" obszarów
#  + generowanie prędkości i czasu przejazdu
#  + wypisywanie do JSON-a