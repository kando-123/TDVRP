import math
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.widgets import Button

graph = dict()

def generate_centroids(count: int) -> list[ tuple[ float, float ] ]:
    # The centroids should be placed in more or less equal angular distances.
    angle = math.tau / count
    base = np.random.rand() * math.tau
    centroids = [ ]
    for i in range(count):
        # Generate the phase
        phase_mean = base + i * angle
        phase_sdev = angle / 6
        phase = np.random.normal(phase_mean, phase_sdev, 1)[0]
        # Generate the radius
        RADIUS_MEAN = 0.50
        RADIUS_SDEV = 0.16
        radius = np.random.normal(RADIUS_MEAN, RADIUS_SDEV, 1)[0]
        radius = max(radius, 0)
        radius = min(radius, 1)
        # Translate to rectangular coordinates
        x, y = radius * math.cos(phase), radius * math.sin(phase)
        centroids.append( (x, y) )
    return centroids

def distance_square(p, q):
    return (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1])

def nearest_neighbor_distances(points: list[ tuple[float, float] ]) -> list[float]:
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

def dhondt_min_1(seats: int, votes: list[float]) -> list[int]:
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
    distances = nearest_neighbor_distances(centroids)
    allocation = dhondt_min_1(count, distances)
    for c in range(len(centroids)):
        # Handle a single centroid in this loop...
        x_cen, y_cen = centroids[c][0], centroids[c][1]
        d = distances[c]
        n_ver = allocation[c]
        for _ in range(n_ver):
            # ...and in this one, handle a single vertex
            ph = np.random.rand() * math.tau
            MEAN = 0.50 * d
            SDEV = 0.25 * d
            r = np.random.normal(MEAN, SDEV, 1)[0]
            v = x_cen + r * math.cos(ph), y_cen + r * math.sin(ph)
            vertices.append(v)
    return vertices

BEGIN =  2
END   = 13
VERTICES_PER_CENTROID = 3

def main():
    centroids_series = [ ]
    vertices_series  = [ ]
    for n in range(BEGIN, END):
        centroids = generate_centroids(n)
        vertices  = generate_vertices(int(VERTICES_PER_CENTROID * n), centroids)
        centroids_series.append(centroids)
        vertices_series.append(vertices)

    figure, axes = plot.subplots()
    plot.subplots_adjust(bottom = 0.2)

    def plot_list(i):
        axes.clear()
        xc, yc = zip(*centroids_series[i])
        xv, yv = zip(*vertices_series[i])
        axes.plot(xc, yc, 'r+')
        axes.plot(xv, yv, 'bo')
        axes.set_title(f'Graph for {len(centroids_series[i])} centroids')
        axes.spines['left'].set_position('zero')
        axes.spines['bottom'].set_position('zero')
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.xaxis.set_ticks_position('bottom')
        axes.yaxis.set_ticks_position('left')
        plot.draw()

    plot_list(0)

    class Index:
        def __init__(self, length):
            self.index = 0
            self.length = length
        
        def next(self, event):
            if self.index < self.length - 1:
                self.index += 1
                plot_list(self.index)
        def prev(self, event):
            if self.index > 0:
                self.index -= 1
                plot_list(self.index)

    callback = Index(len(centroids_series))
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