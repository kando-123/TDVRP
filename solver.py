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
from abc import abstractmethod
from queue import Queue, PriorityQueue

# Element of the graph
class Vertex:
    def __init__(self, coords, order):
        self.coords = coords
        self.order = order
        self.nbors = { }   # Neighbor key => travel times list
        
    def join(self, nbor, weights):
        self.nbors[nbor] = weights
    
    def time(self, nbor, start_time, interval: int) -> float:
        travel_times = self.nbors[nbor]
        return travel_times[int(int(start_time) // int(interval))]

recent_modification = None

class Cluster:
    def __init__(self, v_key, task):
        self.vertices = [v_key]
        self.max_load = task.max_load
        self.graph = task.graph
        self.load = self.graph[v_key].order
    
    def mergeable(self, cluster) -> bool:
        return self.load + cluster.load <= self.max_load
    
    def incorporate(self, cluster) -> bool:
        self.vertices.extend(cluster.vertices)
        self.load += cluster.load
        cluster.vertices.clear( )
        cluster.load = 0
    
    def minimum_metric(lhs, rhs) -> float:
        min_dist = float('inf')
        for k1 in lhs.vertices:
            (x1, y1) = self.graph[k1].coords
            for k2 in rhs.vertices:
                (x2, y2) = self.graph[k2].coords
                dist = math.hypot(x2 - x1, y2 - y1)
                if dist < min_dist:
                    min_dist = dist
        return min_dist
    
    def maximum_metric(lhs, rhs) -> float:
        max_dist = float('-inf')
        for k1 in lhs.vertices:
            (x1, y1) = lhs.graph[k1].coords
            for k2 in rhs.vertices:
                (x2, y2) = rhs.graph[k2].coords
                dist = math.hypot(x2 - x1, y2 - y1)
                if dist > max_dist:
                    max_dist = dist
        return max_dist
    
    def centroid_metric(lhs, rhs) -> float:
        (x1, y1) = (0, 0)
        for k1 in lhs.vertices:
            (x, y) = self.graph[k1].coords
            x1 += x
            y1 += y
        x1 /= len(lhs.vertices)
        y1 /= len(lhs.vertices)
        (x2, y2) = (0, 0)
        for k2 in lhs.vertices:
            (x, y) = self.graph[k1].coords
            x2 += x
            y2 += y
        x2 /= len(rhs.vertices)
        y2 /= len(rhs.vertices)
        return math.hypot(x2 - x1, y2 - y1)
    
    def weighted_metric(lhs, rhs) -> float:
        (x1, y1) = (0, 0)
        for k in lhs.vertices:
            (x, y) = self.graph[k].coords
            q = self.graph[k].order
            x1 += x * q
            y1 += y * q
        x1 /= len(lhs.load)
        y1 /= len(lhs.load)
        (x2, y2) = (0, 0)
        for k in lhs.vertices:
            (x, y) = self.graph[k].coords
            q = self.graph[k].order
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

class Task:
    def __init__(self, source):
        with open(source, 'r') as file:
            data = json.load(file)
        self.graph = dict( )
        for key, val in data["vertices"].items( ):
            coords = (val["x"], val["y"])
            order = val["q"]
            self.graph[key] = Vertex(coords, order)
        for edge in data["edges"]:
            u, v, w = edge["u"], edge["v"], edge["w"]
            self.graph[u].join(v, w)
        fleet = data["fleet"]
        self.n_vehicles = fleet["n_vehicles"]
        self.max_load = fleet["max_load"]
        self.depot = data.get("depot", "0")
        self.interval = data["interval"]
    
    # Returns intermediate nodes and time of arrival to the end
    def connect(self, start: str, end: str, start_time):
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
            for v_nbor in self.graph[v].nbors:
                if v_nbor not in visited:
                    t_nbor = t + self.graph[v].time(v_nbor, t, self.interval)
                    if t_nbor < arrival_time.get(v_nbor, math.inf):
                        predecessor[v_nbor] = v
                        arrival_time[v_nbor] = t_nbor
                    else:
                        t_nbor = arrival_time[v_nbor]
                    queue.put((t_nbor, v_nbor))
        raise Exception('The path was not found')
    
    # Connects successive customers on the path
    def fill(self, path: list[tuple[str, float]], start_time):
        full_path = [ path[0] ]
        for i in range(1, len(path)):
            nodes, start_time = self.connect(path[i-1][0], path[i][0], start_time)
            full_path.extend(nodes)
            full_path.append(path[i])
        return full_path, start_time   # Actually, it's become the end time :) 
    
    def clusterization_solution(self):
        clusters = [ Cluster(key, self) for key in self.graph if key != self.depot ]
        while len(clusters) > self.n_vehicles:
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
            path = [ (self.depot, 0) ]
            while len(cluster.vertices) > 0:
                (x0, y0) = self.graph[path[-1][0]].coords
                min_dist, min_idx = float('inf'), None
                for idx, vtx in enumerate(cluster.vertices):
                    (x1, y1) = self.graph[vtx].coords
                    dist = math.hypot(x1 - x0, y1 - y0)
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = idx
                next = cluster.vertices.pop(min_idx)
                node = (next, self.graph[next].order)
                path.append(node)
            path.append((self.depot, 0))
            path, time = self.fill(path, 0)
            paths.append(path)
            loads.append(cluster.load)
            times.append(time)
        return paths, loads, times
    
    def greedy_solution(self):
        sol = [ ]
        load = [ ]
        time = [ ]
        for _ in range(self.n_vehicles):
            sol.append([(self.depot, 0)])
            load.append(0)
        vertices = { k: v.coords for k, v in self.graph.items( ) if k != self.depot }
        counter = 1000
        while len(vertices) > 0 and counter:
            vehicles = list(range(self.n_vehicles))
            random.shuffle(vehicles)
            for i in vehicles:
                path = sol[i]
                queue = PriorityQueue( )
                (x0, y0) = self.graph[path[-1][0]].coords
                for key, (x, y) in vertices.items( ):
                    dist = math.hypot(x0 - x, y0 - y)
                    queue.put((dist, key))
                while not queue.empty( ):
                    _, key = queue.get( )
                    order = self.graph[key].order
                    if load[i] + order <= self.max_load:
                        path.append((key, order))
                        load[i] += order
                        del vertices[key]
                        break
            counter -= 1
        if counter > 0:   # The loop ended normally
            for path in sol:
                path.append((self.depot, 0))
            for i, path in enumerate(sol):
                sol[i], dur = self.fill(path, 0)
                time.append(dur)
            return sol, load, time
        else:
            raise Exception('Initial solution generation failed')
    
    def failover_solution(self):
        sol = [ ]
        load = [ ]
        time = [ ]
        for _ in range(self.n_vehicles):
            sol.append( [ (self.depot, 0) ] )
            load.append(0)
        vertices = PriorityQueue( )   # Greatest order first
        for k, v in self.graph.items( ):
            if k != self.depot:
                vertices.put((-v.order, k))
        while not vertices.empty( ):
            _, vertex = vertices.get( )
            order = self.graph[vertex].order
            vehicles = PriorityQueue( )
            for i in range(self.n_vehicles):
                vehicles.put( (load[i], i) )   # Least loaded first
            _, vehicle = vehicles.get( )
            if load[vehicle] + order > self.max_load:
                raise Exception('Initial solution generation failed')
            else:
                load[vehicle] += order
                sol[vehicle].append((vertex, order))
        for path in sol:
            path.append((self.depot, 0))
        for i, path in enumerate(sol):
            sol[i], dur = self.fill(path, 0)
            time.append(dur)
        return sol, load, time
    
    def evaluate_path(self, path):
        travel_time = 0
        for i in range(1, len(path)):
            u, v = path[i-1][0], path[i][0]
            travel_time += self.graph[u].time(v, travel_time, self.interval)
        return travel_time
    
    def evaluate_solution(self, solution):
        total_time = 0
        for path in solution:
            total_time += self.evaluate_path(path)
        return total_time
    
    def extract_customers(path):
        return [ node for node in path if node[1] > 0 ]
    
    # Swap two customers on the path
    def swap(self, path: list[tuple[str, float]]):
        customers = Task.extract_customers(path)
        node_idx1 = random.randrange(0, len(customers))
        node_idx2 = random.randrange(1, len(customers))
        if node_idx1 == node_idx2:
            node_idx2 = 0
        customers[node_idx1], customers[node_idx2] = customers[node_idx2], customers[node_idx1]
        # Add the depot at the endpoints
        customers.insert(0, (self.depot, 0))
        customers.append((self.depot, 0))
        path, _ = self.fill(customers, 0)
        return path
    
    # Exchange customers between two paths
    def exchange(self, path1: list[tuple[str, float]], path2: list[tuple[str, float]]):
        customers1 = Task.extract_customers(path1)
        customers2 = Task.extract_customers(path2)
        node_idx1 = random.randrange(0, len(customers1))
        node_idx2 = random.randrange(0, len(customers2))
        customers1[node_idx1], customers2[node_idx2] = customers2[node_idx2], customers1[node_idx1]
        # Add the depot at the endpoints
        customers1.insert(0, (self.depot, 0))
        customers1.append((self.depot, 0))
        customers2.insert(0, (self.depot, 0))
        customers2.append((self.depot, 0))
        path1, _ = self.fill(customers1, 0)
        path2, _ = self.fill(customers2, 0)
        return path1, path2
        # To do: pay attention to max load!
    
    # Move a customer within the path
    def move(self, path: list[tuple[str, float]]):
        customers = Task.extract_customers(path)
        node_idx1 = random.randrange(0, len(customers))
        node = customers.pop(node_idx1)
        node_idx2 = random.randrange(1, len(customers))
        if node_idx1 == node_idx2:
            node_idx2 = 0
        customers.insert(node_idx2, node)
        # Add the depot at the endpoints
        customers.insert(0, (self.depot, 0))
        customers.append((self.depot, 0))
        path, _ = self.fill(customers, 0)
        return path
    
    # Transfer a customer to another path
    def transfer(self, path1: list[tuple[str, float]], path2: list[tuple[str, float]]):
        customers1 = Task.extract_customers(path1)
        customers2 = Task.extract_customers(path2)
        node_idx1 = random.randrange(0, len(customers1))
        node = customers1.pop(node_idx1)
        node_idx2 = random.randrange(0, len(customers2))   # How about selecting the closest path?
        customers2.insert(node_idx2, node)
        # Add the depot at the endpoints
        customers1.insert(0, (self.depot, 0))
        customers1.append((self.depot, 0))
        customers2.insert(0, (self.depot, 0))
        customers2.append((self.depot, 0))
        path1, _ = self.fill(customers1, 0)
        path2, _ = self.fill(customers2, 0)
        return path1, path2
    
    # Reverse a range of customers
    def reverse(self):
        pass
    
    def neighboring_solution(self, solution):
        copy = [ path[:] for path in solution ]   # Deep copy (one-level deep, but that's deep enough)
        [strategy] = random.choices(['swap', 'exchange', 'move', 'transfer'],
            weights=[1, 3, 1, 7], k=1)
        if strategy == 'swap':
            idx = random.randrange(len(copy))
            copy[idx] = self.swap(copy[idx])
        elif strategy == 'exchange':
            idx1, idx2 = random.randrange(0, len(copy)), random.randrange(1, len(copy))
            if idx1 == idx2:
                idx2 = 0
            path1, path2 = self.exchange(copy[idx1], copy[idx2])
            copy[idx1] = path1
            copy[idx2] = path2
        elif strategy == 'move':
            idx = random.randrange(0, len(copy))
            copy[idx] = self.move(copy[idx])
        elif strategy == 'transfer':
            idx1, idx2 = random.randrange(0, len(copy)), random.randrange(1, len(copy))
            if idx1 == idx2:
                idx2 = 0
            path1, path2 = self.transfer(copy[idx1], copy[idx2])
            copy[idx1] = path1
            copy[idx2] = path2
        else:
            raise Exception('Something weird has happened')
        global recent_modification
        recent_modification = strategy
        return copy
    
    def edge_time(u, v, t0):
        return self.graph[u].time(v, t0, self.interval)

def main( ):
    task = Task(sys.argv[1])
    
    paths, loads, times = None, None, None
    
    try:
        paths, loads, times = task.clusterization_solution( )
    except:
        for _ in range(10):
            try:
                paths, loads, times = task.greedy_solution( )
                print('break')
                break
            except Exception as e:
                print(e)
                print('exception!')
                pass
        else:
            try:
                paths, loads, times = task.failover_solution( )
            except:
                print('Failure')
                exit(1)
    
    # The algorithm for determining the initial solution can be upgraded later.
    # Nevertheless, at this point there is SOME initial solution to start with.
    
    N_ITER = 10_000
    TEMP   =    100
    best_sol, best_eval = paths, task.evaluate_solution(paths)
    curr_sol, curr_eval = best_sol, best_eval
    scores = [best_eval]
    
    for i in range(N_ITER):
        t = TEMP / float(i + 1)
        
        cand_sol  = task.neighboring_solution(paths)
        cand_eval = task.evaluate_solution(cand_sol)
        
        if cand_eval < best_eval or random.random( ) < math.exp((curr_eval - cand_eval) / t):
            curr_sol, curr_eval = cand_sol, cand_eval
            if cand_eval < best_eval:
                print(f'Upgrade, {recent_modification}!')
                best, best_eval = cand_sol, cand_eval
                scores.append(best_eval)
        
        if i % 100 == 0:
            print(f"Iteration {i}")
    
    best_sol = [ [ list(node) for node in path ] for path in best_sol ]
    print(best_sol)
    print(best_eval)
    
main( )