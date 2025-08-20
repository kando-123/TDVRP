# Load the graph.
# Determine the initial solution.
# In a loop:
#     Determine a neighboring solution.
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
from queue import PriorityQueue

class Vertex:
    def __init__(self, coords, order):
        self.coords = coords
        self.order = order
        self.nbors = { }   # Neighbor key => travel times list
        
    def join(self, nbor, weights):
        self.nbors[nbor] = weights
    
    def time(self, nbor, start_time, period=None):
        travel_times = self.nbors[nbor]
        if period is None:
            period = 86400 / len(travel_times)    # Assume the list of travel times covers whole day
        return travel_times[int(int(start_time) // int(period))]

class Problem:
    def __init__(self, source):
        with open(source, 'r') as file:
            data = json.load(file)
        self.graph = dict( )
        for key, val in data['vertices'].items( ):
            coords = (val['x'], val['y'])
            order = val.get('q', 0)
            self.graph[key] = Vertex(coords, order)
        for edge in data['edges']:
            u, v, w = edge['u'], edge['v'], edge['w']
            self.graph[u].join(v, w)
        fleet = data['fleet']
        self.n_vehicles = fleet['size']
        self.max_load = fleet['max_load']
        self.depot = data.get('depot', '0')
    
    def fill(path):
        
    
    def failover_solution(self):
        sol = [ ]
        load = [ ]
        for _ in range(self.n_vehicles):
            sol.append([self.depot])
            load.append(0)
        customer_vertices = PriorityQueue( )   # Greatest order first
        for k, v in self.graph.items( ):
            if v.order > 0:
                customer_vertices.put( (-v.order, k) )
        while not customer_vertices.empty( ):
            _, vertex = customer_vertices.get( )
            order = self.graph[vertex].order
            vehicles = PriorityQueue( )
            for i in range(self.n_vehicles):
                vehicles.put( (load[i], i) )   # Least loaded first
            _, vehicle = vehicles.get( )
            if load[vehicle] + order > self.max_load:
                raise Exception('Initial solution generation failed')
            else:
                load[vehicle] += order
                sol[vehicle].append(vertex)
        return sol, load
    
    def initial_solution(self, n_tries=10):
        sol = [ ]
        load = [ ]
        for _ in range(self.n_vehicles):
            sol.append([self.depot])
            load.append(0)
        customer_vertices = { k: v.coords for k, v in self.graph.items( ) if v.order > 0 }
        safety_counter = 1000
        while len(customer_vertices) > 0 and safety_counter:
            vehicles = list(range(self.n_vehicles))
            random.shuffle(vehicles)
            for i in vehicles:
                path = sol[i]
                queue = PriorityQueue( )
                (x0, y0) = self.graph[path[-1]].coords
                for key, (x, y) in customer_vertices.items( ):
                    dist = math.hypot(x0 - x, y0 - y)
                    queue.put((dist, key))
                while not queue.empty( ):
                    _, key = queue.get( )
                    order = self.graph[key].order
                    if load[i] + order <= self.max_load:
                        path.append(key)
                        load[i] += order
                        del customer_vertices[key]
                        break
            safety_counter -= 1
        if safety_counter > 0:   # The loop ended normally
            for path in sol:
                path.append(self.depot)
            return sol, load
        else:
            raise Exception('Initial solution generation failed')
    
    def neighboring_solution(self, solution):
        raise NotImplementedError( )

def edge_time(graph, u, v, t0, period=None):
    return graph[u].time(v, t0, period)

def main( ):
    problem = Problem(sys.argv[1])
    
    sol, load = None, None
    for _ in range(10):
        try:
            sol, load = problem.initial_solution( )
            break
        except:
            pass
    else:
        try:
            sol, load = problem.failover_solution( )
        except:
            print('Failure')
            exit(1)
    
    print(sol)
    print(load)
    
    # vertices = { k: (v.x, v.y) for k, v in graph.items( ) if v.q > 0 }
    # clusters = clusterize(vertices, 4)
    # for i, c in enumerate(clusters):
    #     print(f'Cluster {i}:', end='\n\t')
    #     for k in c.keys( ):
    #         print(k, end=' ')
    #     print(f': { sum( graph[k].q for k in c.keys( ) ) }')
    #     # print( )
    
main( )