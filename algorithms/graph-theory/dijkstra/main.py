import heapq as hq

labels = ['s', 't', 'y', 'x', 'z']
graph = [             # Nodes
#    s  t  y  x  z
    [0, 3, 5, 0, 0],  # s
    [0, 0, 2, 6, 0],  # t
    [0, 1, 0, 4, 6],  # y
    [0, 0, 0, 0, 2],  # x
    [3, 0, 0, 7, 0]   # z
]
INFINITY = float('inf')


# g - adjaceny list of weighted graph
# n - the number of nodes in the graph
# s - the index of the starting node (0 <= s <= n)

def dijkstra(g, n, s):
    visited = [False] * n  # easy way to show that if we have visited a node
    dist = [INFINITY] * n
    pred = [None] * n
    dist[s] = 0
    queue = []
    hq.heappush(queue, (s, 0))
    while len(queue) != 0:
        index, min = queue.pop()
        visited[index] = True
        for edge in range(len(g[index])):
            if visited[edge]: continue
            if g[index][edge] == 0: continue
            newDist = dist[index] + g[index][edge]
            if newDist < dist[edge]:
                pred[edge] = labels[index]
                dist[edge] = newDist
                hq.heappush(queue, (edge, newDist))
            print(dist)
    print(f'predecessor {pred}')
    return dist

print('------------------- Testing ----------------- ')
print('distance for index 0 to all vs')
print(dijkstra(graph, 5, 0)) # starting v -> index 0
print('---------------------------------------------')
print('distances for index 4 to all vs')
print(dijkstra(graph, 5, 4)) # starting v -> index 4

