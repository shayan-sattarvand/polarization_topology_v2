import time


class DisjointSetUnion:
    def __init__(self, n):
        self.disjoint_set = list(range(n))
        self.set_sizes = [1] * n
        self.connected_components = n
    
    def find(self, x):
        if x == self.disjoint_set[x]: return x
        self.disjoint_set[x] = self.find(self.disjoint_set[x])
        return self.disjoint_set[x]

    def union(self, u, v):
        a = self.find(u)
        b = self.find(v)
        if a == b: return
        if self.set_sizes[a] < self.set_sizes[b]: a, b = b, a
        self.disjoint_set[b] = a
        self.set_sizes[a] += self.set_sizes[b]
        self.set_sizes[b] = 0
        self.connected_components -= 1

    def query(self, u, v):
        return self.find(u) == self.find(v)
    
    def connected_components_edges(self):
        a = [i for i in self.set_sizes if i > 1]
        return len(a)



def print_bin(x):
    print(f'{x:b}')


def cc_density(binary_representation):
    """
    Calculate the CC density for a given binary representation of a configuration.
    
    Parameters:
    binary_representation (str): A string of '0's and '1's representing the configuration.
    
    Returns:
    float: The calculated CC density.
    """
    dsu = DisjointSetUnion(5)
    inter_wheel = binary_representation % 4
    wheel_rep = binary_representation // 4
    binary_representation = binary_representation // 4
    rim_connections = binary_representation % 16
    binary_representation = binary_representation // 16
    wheel_connections = binary_representation
    for i in range(4):
        if rim_connections & (1 << (3 - i)):
            dsu.union(i + 1, (i + 1) % 4 + 1)
    for i in range(4):
        if wheel_connections & (1 << (3 - i)):
            dsu.union(0, i + 1)
    inter_wheel_new = inter_wheel
    if (inter_wheel & 2 and dsu.query(1,3)) or (inter_wheel & 1 and dsu.query(2,4)):
        center_loc = dsu.find(0)  # gonna be same as the path location cuz it has to go thru center
        for i in range(4):
            loc = dsu.find(i + 1)
            print_bin(wheel_rep)

            if loc == center_loc and wheel_rep & (1 << (7 - i)):
                wheel_rep -= 1 << (7 - i)
            loc2 = dsu.find((i + 1) % 4 + 1)
            if loc2 == center_loc and loc == center_loc and wheel_rep & (1 << (3 - i)):
                wheel_rep -= 1 << (3 - i)
        if inter_wheel & 2 and dsu.query(1,3):
            inter_wheel_new -= 2
        if inter_wheel & 1 and dsu.query(2,4):
            inter_wheel_new -= 1
    dsu_now = DisjointSetUnion(5)
    for i in range(4):
        if wheel_rep & (1 << (3 - i)):
            dsu_now.union(i + 1, (i + 1) % 4 + 1)
    for i in range(5):
        dsu_now.find(i)
    for i in range(4):
        if wheel_rep & (1 << (7 - i)):
            dsu_now.union(0, i + 1)
    for i in range(5):
        dsu_now.find(i)
    for i in range(2):
        if inter_wheel_new & (1 << (1 - i)):
            dsu_now.union(i + 1, i + 3)
    for i in range(5):
        dsu_now.find(i)
    return dsu_now.connected_components_edges()

if __name__ == '__main__':
    import fig_hexagonal
    DEG = fig_hexagonal.DEG

    

