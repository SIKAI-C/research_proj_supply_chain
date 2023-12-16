import matplotlib.pyplot as plt


def fixPartitions(x):
    edges = {(i,j): int(x[i,j]) for i,j in x.keys() if x[i,j] > 0.5}
    tours = []
    all_edges = list(edges.keys())
    while all_edges:
        cycle = []
        start = all_edges[0][0]
        next = all_edges[0][1]
        cycle.append(start)
        cycle.append(next)
        all_edges.remove((start, next))
        while next != start:
            for i in range(len(all_edges)):
                if all_edges[i][0] == next:
                    next = all_edges[i][1]
                    cycle.append(next)
                    all_edges.remove(all_edges[i])
                    break
        tours.append(cycle)
    return tours


def solJoint(s):
    res = [0]
    while True:
        for i,j in s:
            if i == res[-1]:
                res.append(j)
                break
        if res[-1] == 0: break
    return res


def actualRoutes(y, K, T):
    sol = [[[] for _ in range(len(K))] for _ in range(len(T))]
    for i,j,k,t in y.keys():
        if y[i,j,k,t] > 0:
            sol[t-1][k-1].append((i,j))

    for i in range(len(T)):
        for j in range(len(K)):
            sol[i][j] = solJoint(sol[i][j])

    for i in range(len(T)):
        print("Day", i+1)
        for j in range(len(K)):
            print("    Car", j+1, sol[i][j])
    
    return sol


def plotRoutes(coordinates, partitions, routes):

    colors = ['g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple',\
            'brown', 'pink', 'gray', 'olive', 'cyan', 'lime', 'teal', 'lavender',\
            'maroon', 'navy', 'gold', 'salmon', 'tan', 'aqua', 'indigo', 'azure',\
            'beige', 'coral', 'crimson', 'fuchsia', 'honeydew', 'ivory', 'khaki',\
            'linen', 'magenta', 'mint', 'plum', 'snow', 'thistle', 'wheat', 'yellowgreen']
    
    T = len(routes)
    dimension = len(coordinates)
    
    for n, route in enumerate(partitions):
        fig = plt.figure(figsize=(5,5))
        for i in range(dimension):
            if i == 0: plt.scatter(coordinates[i][0], coordinates[i][1], c='red', alpha=0.5, s=300)
            else: plt.scatter(coordinates[i][0], coordinates[i][1], c='black', alpha=0.5, s=100)
            plt.text(coordinates[i][0], coordinates[i][1], str(i), fontsize=10)
        for i in range(len(route)-1):
            plt.plot([coordinates[route[i]][0], coordinates[route[i+1]][0]],
                    [coordinates[route[i]][1], coordinates[route[i+1]][1]], c='red', alpha=0.5)
        plt.title("Partition " + str(n))
        plt.show()

    for t in range(T):
        fig = plt.figure(figsize=(5,5))
        for i in range(dimension):
            if i == 0: plt.scatter(coordinates[i][0], coordinates[i][1], c='red', alpha=0.5, s=300)
            else: plt.scatter(coordinates[i][0], coordinates[i][1], c='black', alpha=0.5, s=100)
            plt.text(coordinates[i][0], coordinates[i][1], str(i), fontsize=10)

        for route in partitions:
            for i in range(len(route)-1):
                plt.plot([coordinates[route[i]][0], coordinates[route[i+1]][0]],
                        [coordinates[route[i]][1], coordinates[route[i+1]][1]], c='red', alpha=0.2)
        
        for n, route in enumerate(routes[t]):
            for i in range(len(route)-1):
                plt.plot([coordinates[route[i]][0], coordinates[route[i+1]][0]],
                        [coordinates[route[i]][1], coordinates[route[i+1]][1]], c=colors[n])
        
        plt.title("Day " + str(t+1))
        plt.show()