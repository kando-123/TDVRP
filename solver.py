# Load the graph.
# Determine the initial solution.
# In a loop:
#     Determine a neighboring solution
#     Compare
#     Swap conditionally
#
# Determine a neighboring solution:
#     (1) The change must produce a valid path
#     (2) If a customer node is being included, it must be also excluded from another path.
#         * Actually, it should NOT be forbidden to pass through a customer vertex without servicing it.

import json
import math
import random
import sys
from queue import Queue, PriorityQueue

class Task:
    graph = None
    n_vehicles = None
    max_load = None
    depot = None
    interval = None

# Element of the graph
class Vertex:
    def __init__(self, coords, order):
        self.coords = coords
        self.order = order
        self.nbors = { }   # Neighbor key => travel times list
        
    def join(self, nbor, weights):
        self.nbors[nbor] = weights
    
    def travel_time(self, nbor, start_time) -> float:
        travel_times = self.nbors[nbor]
        return travel_times[int(int(start_time) // int(Task.interval))]

class Cluster:
    def __init__(self, v_key):
        self.vertices = [v_key]
        self.load = Task.graph[v_key].order
    
    def mergeable(self, cluster) -> bool:
        return self.load + cluster.load <= Task.max_load
    
    def incorporate(self, cluster) -> bool:
        self.vertices.extend(cluster.vertices)
        self.load += cluster.load
        cluster.vertices.clear( )
        cluster.load = 0
    
    def minimum_metric(lhs, rhs) -> float:
        min_dist = float('inf')
        for k1 in lhs.vertices:
            (x1, y1) = Task.graph[k1].coords
            for k2 in rhs.vertices:
                (x2, y2) = Task.graph[k2].coords
                dist = math.hypot(x2 - x1, y2 - y1)
                if dist < min_dist:
                    min_dist = dist
        return min_dist
    
    def maximum_metric(lhs, rhs) -> float:
        max_dist = float('-inf')
        for k1 in lhs.vertices:
            (x1, y1) = Task.graph[k1].coords
            for k2 in rhs.vertices:
                (x2, y2) = Task.graph[k2].coords
                dist = math.hypot(x2 - x1, y2 - y1)
                if dist > max_dist:
                    max_dist = dist
        return max_dist
    
    def centroid_metric(lhs, rhs) -> float:
        (x1, y1) = (0, 0)
        for k1 in lhs.vertices:
            (x, y) = Task.graph[k1].coords
            x1 += x
            y1 += y
        x1 /= len(lhs.vertices)
        y1 /= len(lhs.vertices)
        (x2, y2) = (0, 0)
        for k2 in lhs.vertices:
            (x, y) = Task.graph[k1].coords
            x2 += x
            y2 += y
        x2 /= len(rhs.vertices)
        y2 /= len(rhs.vertices)
        return math.hypot(x2 - x1, y2 - y1)
    
    def weighted_metric(lhs, rhs) -> float:
        (x1, y1) = (0, 0)
        for k in lhs.vertices:
            (x, y) = Task.graph[k].coords
            q = Task.graph[k].order
            x1 += x * q
            y1 += y * q
        x1 /= len(lhs.load)
        y1 /= len(lhs.load)
        (x2, y2) = (0, 0)
        for k in lhs.vertices:
            (x, y) = Task.graph[k].coords
            q = Task.graph[k].order
            x2 += x * q
            y2 += y * q
        x2 /= len(rhs.load)
        y2 /= len(rhs.load)
        return math.hypot(x2 - x1, y2 - y1)
    
    metric = maximum_metric
    
    def set_metric(str):
        if str == 'minimum' or str == 'min':
            metric = minimum_metric
        elif str == 'maximum' or str == 'max':
            metric = maximum_metric
        elif str == 'centroid':
            metric = centroid_metric
        elif str == 'weighted':
            metric = centroid_metric
        else:
            raise ValueError('Wrong metric name')
    
    def distance(self, other) -> float:
        return Cluster.metric(self, other)

def init(source):
    Task.graph = dict( )
    with open(source, 'r') as file:
        data = json.load(file)
    for key, val in data["vertices"].items( ):
        coords = (val["x"], val["y"])
        order = val["q"]
        Task.graph[key] = Vertex(coords, order)
    for edge in data["edges"]:
        u, v, w = edge["u"], edge["v"], edge["w"]
        Task.graph[u].join(v, w)
    fleet = data["fleet"]
    
    Task.n_vehicles = fleet["n_vehicles"]
    Task.max_load   = fleet["max_load"]
    Task.depot      = data.get("depot", "0")
    Task.interval   = data["interval"]

class Path:
    def find_intermediate_vertices(start: str, end: str, start_time):
        if start == end:
            return [ ], start_time
        predecessor  = { start: None }
        arrival_time = { start: start_time }
        visited = set( )
        queue = PriorityQueue( )
        queue.put((start_time, start))
        while not queue.empty( ):
            _, v = queue.get( )
            if v == end:
                path = [ ]
                vertex = predecessor[end]
                while vertex != start and end is not None:
                    path.insert(0, (vertex, 0))
                    vertex = predecessor[vertex]
                return path, arrival_time[end]
            visited.add(v)
            t = arrival_time[v]
            for v_nbor in Task.graph[v].nbors:
                if v_nbor not in visited:
                    t_nbor = t + Task.graph[v].travel_time(v_nbor, t)
                    if t_nbor < arrival_time.get(v_nbor, math.inf):
                        predecessor[v_nbor] = v
                        arrival_time[v_nbor] = t_nbor
                    else:
                        t_nbor = arrival_time[v_nbor]
                    queue.put((t_nbor, v_nbor))
        raise Exception('The path was not found')
    
    def connect(nodes: list[tuple[str, float]], start_time):
        full_path = [nodes[0]]
        for i in range(1, len(nodes)):
            intermediates, start_time = Path.find_intermediate_vertices(nodes[i-1][0], nodes[i][0], start_time)
            full_path.extend(intermediates)
            full_path.append(nodes[i])
        return full_path
    
    def __init__(self, vertices: list[str]):
        self.nodes = Path.connect([(v, Task.graph[v].order) for v in vertices], 0)
    
    def extract_customers(self):
        return [ node for node in self.nodes if node[1] > 0 ]
    
    def copy(self):
        vertices = [Task.depot] + [vtx for vtx, ord in self.extract_customers( )] + [Task.depot]
        return Path(vertices)
    
    def load(self):
        return sum([Task.graph[vtx].order for vtx, ord in self.nodes])
    
    def travel_time(self):
        time = 0
        for i in range(1, len(self.nodes)):
            prev = self.nodes[i-1][0]
            next = self.nodes[ i ][0]
            time += Task.graph[prev].travel_time(next, time)
        return time
    
    # ----- Internal Modifications ----- #
    
    def shuffle(self, length):
        customers = self.extract_customers( )
        if length < len(customers):
            first = random.randint(0, len(customers) - length)
            last = first + length
            slice = customers[first:last]
            random.shuffle(slice)
            if len(slice) > 1:
                i = 5
                while slice == customers[first:last] and i > 0:
                    random.shuffle(slice)
                    i -= 1
            customers = customers[0:first] + slice + customers[last:]
            customers.insert(0, (Task.depot, 0))
            customers.append((Task.depot, 0))
            self.nodes = Path.connect(customers, 0)
        else:
            random.shuffle(self.nodes)
    
    def reverse(self, length):
        customers = self.extract_customers( )
        if length < len(customers):
            first = random.randint(0, len(customers) - length)
            last = first + length
            slice = customers[first:last]
            slice.reverse( )
            customers = customers[:first] + slice + customers[last:]
            customers.insert(0, (Task.depot, 0))
            customers.append((Task.depot, 0))
            self.nodes = Path.connect(customers, 0)
        else:
            self.nodes.reverse( )
    
    def move(self, length=1):
        customers = self.extract_customers( )
        if length < len(customers):
            first = random.randint(0, len(customers) - length)
            last = first + length
            sublist = customers[first:last]
            del customers[first:last]
            index = random.randrange(1, len(customers))
            if index == first:
                index = 0
            customers[index:index] = sublist
            customers.insert(0, (Task.depot, 0))
            customers.append((Task.depot, 0))
            self.nodes = Path.connect(customers, 0)

    # ----- Mutual Modifications ----- #
    
    # The mutual modifications rely on extracting sequences of nodes (in particular, single nodes)
    # from one list and inserting them to others.
    
    # Removes the 
    def remove(self, length=1) -> tuple[list[str], float]:
        pass

def clusterization_solution( ):
    clusters = [ Cluster(key) for key in Task.graph if key != Task.depot ]
    while len(clusters) > Task.n_vehicles:
        # Queue will store pairs (i, j) of clusters, i < j,
        #     from the least distant to the most
        queue = PriorityQueue( )
        for i in range(0, len(clusters)):
            c1 = clusters[i]
            for j in range(i + 1, len(clusters)):
                c2 = clusters[j]
                entry = c1.distance(c2), (i, j)
                queue.put(entry)
        # Merge the two closest clusters,
        #    on condition they will not exceed the max load
        while not queue.empty( ):
            _, (i, j) = queue.get( )
            c1 = clusters[i]
            c2 = clusters[j]
            if c1.mergeable(c2):
                c1.incorporate(c2)
                clusters.pop(j)
                break
        else:
            raise Exception('Clusterization failed!')
    # Transform the clusters into paths
    paths = [ ]
    loads = [ ]
    times = [ ]
    for cluster in clusters:
        customers = [ Task.depot ]
        while len(cluster.vertices) > 0:
            (x0, y0) = Task.graph[customers[-1][0]].coords
            min_dist, min_idx = float('inf'), None
            for idx, vtx in enumerate(cluster.vertices):
                (x1, y1) = Task.graph[vtx].coords
                dist = math.hypot(x1 - x0, y1 - y0)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
            next = cluster.vertices.pop(min_idx)
            customers.append(next)
        customers.append(Task.depot)
        path = Path(customers)
        paths.append(path)
        loads.append(path.load( ))
        times.append(path.travel_time( ))
    return paths, loads, times

# def greedy_solution( ):
#     paths = [ ]
#     loads = [ ]
#     times = [ ]
#     for _ in range(Task.n_vehicles):
#         paths.append([(Task.depot, 0)])
#         loads.append(0)
#     vertices = { k: v.coords for k, v in Task.graph.items( ) if k != Task.depot }
#     counter = 1000
#     while len(vertices) > 0 and counter:
#         vehicles = list(range(Task.n_vehicles))
#         random.shuffle(vehicles)
#         for i in vehicles:
#             path = paths[i]
#             queue = PriorityQueue( )
#             (x0, y0) = Task.graph[path[-1][0]].coords
#             for key, (x, y) in vertices.items( ):
#                 dist = math.hypot(x0 - x, y0 - y)
#                 queue.put((dist, key))
#             while not queue.empty( ):
#                 _, key = queue.get( )
#                 order = Task.graph[key].order
#                 if loads[i] + order <= Task.max_load:
#                     paths.append((key, order))
#                     loads[i] += order
#                     del vertices[key]
#                     break
#         counter -= 1
#     if counter > 0:   # The loop ended normally
#         for path in paths:
#             path.append((Task.depot, 0))
#         for i, path in enumerate(paths):
#             paths[i] = Path(path)
#             time.append(paths[i].travel_time( ))
#         return paths, load, time
#     else:
#         raise Exception('Initial solution generation failed')
# 
# def failover_solution( ):
#     paths = [ ]
#     loads = [ ]
#     times = [ ]
#     for _ in range(Task.n_vehicles):
#         paths.append( [ (Task.depot, 0) ] )
#         loads.append(0)
#     vertices = PriorityQueue( )   # Greatest order first
#     for k, v in Task.graph.items( ):
#         if k != Task.depot:
#             vertices.put((-v.order, k))
#     while not vertices.empty( ):
#         _, vertex = vertices.get( )
#         order = Task.graph[vertex].order
#         vehicles = PriorityQueue( )
#         for i in range(Task.n_vehicles):
#             vehicles.put((load[i], i))   # Least loaded first
#         _, vehicle = vehicles.get( )
#         if loads[vehicle] + order > Task.max_load:
#             raise Exception('Initial solution generation failed')
#         else:
#             loads[vehicle] += order
#             paths[vehicle].append((vertex, order))
#     for path in sol:
#         path.append((Task.depot, 0))
#     for i, path in enumerate(sol):
#         paths[i] = Path(path)
#         time.append(paths[i].travel_time( ))
#     return paths, load, time

def evaluate_solution(solution: list[Path]):
    total_time = 0
    for path in solution:
        total_time += path.travel_time( )
    return total_time

def neighboring_solution(solution):
    new_solution = None
    [strategy] = random.choices(['shuffle', 'reverse', 'move'], k=1)
    if strategy == 'shuffle':
        index = random.randrange(len(solution))
        copy = solution[index].copy( )
        copy.shuffle(random.randint(3, 5))
        new_solution = [ ]
        for i, path in enumerate(solution):
            new_solution.append(copy if i == index else path)
    elif strategy == 'reverse':
        index = random.randrange(len(solution))
        copy = solution[index].copy( )
        copy.reverse(random.randint(2, 4))
        new_solution = [ ]
        for i, path in enumerate(solution):
            new_solution.append(copy if i == index else path)
    elif strategy == 'move':
        index = random.randrange(len(solution))
        copy = solution[index].copy( )
        copy.move(random.randint(1, 3))
        new_solution = [ ]
        for i, path in enumerate(solution):
            new_solution.append(copy if i == index else path)
    else:
        raise Exception('Something weird has happened')
    global recent_modification
    recent_modification = strategy
    return new_solution

def edge_time(u, v, t0):
    return Task.graph[u].travel_time(v, t0)

def main( ):
    init(sys.argv[1])
    
    # paths, loads, times = None, None, None
    # 
    # try:
    #     paths, loads, times = clusterization_solution( )
    # except:
    #     for _ in range(10):
    #         try:
    #             paths, loads, times = greedy_solution( )
    #             print('break')
    #             break
    #         except Exception as e:
    #             print(e)
    #             print('exception!')
    #             pass
    #     else:
    #         try:
    #             paths, loads, times = failover_solution( )
    #         except:
    #             print('Failure')
    #             exit(1)
    
    # The algorithm for determining the initial solution can be upgraded later.
    # Nevertheless, at this point there is SOME initial solution to start with.
    
    paths, loads, times = clusterization_solution( )
    
    N_ITER = 100
    TEMP   = 100
    best_sol, best_eval = paths, evaluate_solution(paths)
    curr_sol, curr_eval = best_sol, best_eval
    scores = [best_eval]
    
    for i in range(N_ITER):
        t = TEMP / float(i + 1)
        
        cand_sol  = neighboring_solution(paths)
        cand_eval = evaluate_solution(cand_sol)
        
        if cand_eval < best_eval or random.random( ) < math.exp((curr_eval - cand_eval) / t):
            curr_sol, curr_eval = cand_sol, cand_eval
            if cand_eval < best_eval:
                print(f'Upgrade, {recent_modification}!')
                best, best_eval = cand_sol, cand_eval
                scores.append(best_eval)
        
        if i % 100 == 0:
            print(f"Iteration {i}")
    
    best_sol = [ [ list(node) for node in path.nodes ] for path in best_sol ]
    print(best_sol)
    print(best_eval)
    
main( )