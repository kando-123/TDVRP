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
        return sum([ord for vtx, ord in self.nodes])
    
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
    
    def remove(self, length=1):
        customers = self.extract_customers( )
        if length < len(customers):
            first = random.randint(0, len(customers) - length)
            last = first + length
            sublist = customers[first:last]
            del customers[first:last]
            customers.insert(0, (Task.depot, 0))
            customers.append((Task.depot, 0))
            self.nodes = Path.connect(customers, 0)
            return sublist
        else:
            return None
    
    def insertable(self, new_customers: list[tuple[str, float]]):
        own_load = self.load()
        new_load = sum([ord for vtx, ord in new_customers])
        return own_load + new_load <= Task.max_load
    
    def distance(self, new_customers: list[tuple[str, float]]):
        customers = self.extract_customers( )
        cumulative_distance = 0
        for new_vertex, new_order in new_customers:
            (x0, y0) = Task.graph[new_vertex].coords
            min_dist, min_index = float('inf'), None
            for index in range(1, len(customers)):
                prev, _ = customers[index-1]
                next, _ = customers[index]
                (x1, y1) = Task.graph[prev].coords
                (x2, y2) = Task.graph[next].coords
                dist = math.hypot(x1 - x0, y1 - y0) + math.hypot(x2 - x0, y2 - y0)
                if dist < min_dist:
                    min_dist = dist
                    min_index = index
            cumulative_distance += min_dist
        if cumulative_distance == float('inf'):
            print('Infinitah!')
        return cumulative_distance
    
    def insert(self, new_customers: list[tuple[str, float]]):
        customers = self.extract_customers( )
        for new_vertex, new_order in new_customers:
            (x0, y0) = Task.graph[new_vertex].coords
            min_dist, min_index = float('inf'), None
            for index in range(1, len(customers)):
                prev, _ = customers[index-1]
                next, _ = customers[index]
                (x1, y1) = Task.graph[prev].coords
                (x2, y2) = Task.graph[next].coords
                dist = math.hypot(x1 - x0, y1 - y0) + math.hypot(x2 - x0, y2 - y0)
                if dist < min_dist:
                    min_dist = dist
                    min_index = index
            customers.insert(min_index, (new_vertex, new_order))
        customers.insert(0, (Task.depot, 0))
        customers.append((Task.depot, 0))
        self.nodes = Path.connect(customers, 0)

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
    return paths

def failover_solution( ):
    paths = [ ]
    loads = [ ]
    for _ in range(Task.n_vehicles):
        paths.append([Task.depot])
        loads.append(0)
    vertices = PriorityQueue( )   # Greatest order first
    for k, v in Task.graph.items( ):
        if k != Task.depot:
            vertices.put((-v.order, k))
    while not vertices.empty( ):
        _, vertex = vertices.get( )
        order = Task.graph[vertex].order
        # Least loaded first
        vehicle = 0
        for i in range(1, Task.n_vehicles):
            if loads[i] < loads[vehicle]:
                vehicle = 1
        if loads[vehicle] + order > Task.max_load:
            raise Exception('Initial solution generation failed')
        else:
            loads[vehicle] += order
            paths[vehicle].append(vertex)
    for path in paths:
        path.append(Task.depot)
    return [ Path(path) for path in paths ]


def evaluate_solution(solution: list[Path]):
    total_time = 0
    for path in solution:
        total_time += path.travel_time( )
    return total_time

def shuffle(solution):
    index = random.randrange(len(solution))
    copy = solution[index].copy( )
    copy.shuffle(random.randint(3, 5))
    new_solution = [ ]
    for i, path in enumerate(solution):
        new_solution.append(copy if i == index else path)
    return new_solution

def reverse(solution):
    index = random.randrange(len(solution))
    copy = solution[index].copy( )
    copy.reverse(random.randint(2, 4))
    new_solution = [ ]
    for i, path in enumerate(solution):
        new_solution.append(copy if i == index else path)
    return new_solution

def move(solution):
    index = random.randrange(len(solution))
    copy = solution[index].copy( )
    copy.move(random.randint(1, 3))
    new_solution = [ ]
    for i, path in enumerate(solution):
        new_solution.append(copy if i == index else path)
    return new_solution

def transfer(solution):
    index1 = random.randrange(len(solution))
    copy1 = solution[index1].copy( )
    sequence = copy1.remove(random.randint(1, 2))
    min_dist, index2 = float('inf'), None
    for i, path in enumerate(solution):
        if i == index1:
            continue
        if path.insertable(sequence) and (dist := path.distance(sequence)) < min_dist:
            min_dist = dist
            index2 = i
    if index2 is not None:
        copy2 = solution[index2].copy( )
        copy2.insert(sequence)
        new_solution = [ ]
        for i, path in enumerate(solution):
            if i == index1:
                new_solution.append(copy1)
            elif i == index2:
                new_solution.append(copy2)
            else:
                new_solution.append(path)
    else:
        new_solution = solution
    return new_solution

def exchange(solution):
    # Index 1
    index1 = random.randrange(0, len(solution))
    copy1 = solution[index1].copy( )
    sequence1 = copy1.remove(random.randint(1, 2))
    # Index 2
    index2 = random.randrange(1, len(solution))
    if index2 == index1:
        index2 = 0
    copy2 = solution[index2].copy( )
    sequence2 = copy1.remove(random.randint(1, 2))
    # Exchange (if possible)
    if copy1.insertable(sequence2) and copy2.insertable(sequence1):
        copy1.insert(sequence2)
        copy2.insert(sequence1)
        # Solution
        new_solution = [ ]
        for i, path in enumerate(solution):
            if i == index1:
                new_solution.append(copy1)
            elif i == index2:
                new_solution.append(copy2)
            else:
                new_solution.append(path)
    else:
        # Failover solution
        new_solution = solution
    return new_solution

def neighboring_solution(solution):
    new_solution = None
    # [strategy] = random.choices([shuffle, reverse, move, transfer, exchange], weights=[10, 10, 10, 1, 1], k=1)
    message, strategy = random.choices([
            ('shuffle', shuffle),
            ('reverse', reverse),
            ('move', move),
            ('transfer', transfer),
            ('exchange', exchange)
        ], k=1)[0]
    print(message)
    new_solution = strategy(solution)
    return new_solution

def main(in_file, out_file):
    init(in_file)
    
    try:
        paths = clusterization_solution( )
    except:
        try:
            paths = failover_solution( )
        except Exception as e:
            print(e)
            exit(1)
    print('Initial solution generation: success')
    
    N_ITER = 5000
    TEMP   =  500
    best, best_cost = paths, evaluate_solution(paths)
    current, current_cost = best, best_cost
    
    ALPHA = TEMP/N_ITER
    for i in range(N_ITER):
        # t = TEMP / float(i + 1)
        t = TEMP - ALPHA * i
        # t = TEMP / math.log(1 + i)
        
        candidate      = neighboring_solution(paths)
        candidate_cost = evaluate_solution(candidate)
        
        if (better := candidate_cost < best_cost) or random.random( ) < math.exp((current_cost - candidate_cost) / t):
            current, current_cost = candidate, candidate_cost
            if better:
                print(f'Upgrade')
                best, best_cost = candidate, candidate_cost
        
        if i % 10 == 0:
            print(f"Iteration {i}, temperature: {t:.3f}")
    
    best = [ [ list(node) for node in path.nodes ] for path in best ]
    print(best)
    print(best_cost)
    
    with open(out_file, 'w') as file:
        json.dump(best, file)

if len(sys.argv) < 3:
    print('USAGE: python solver.py <in: graph.json> <out: solution.json>')
else:
    main(sys.argv[1], sys.argv[2])