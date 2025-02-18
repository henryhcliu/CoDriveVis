class UnionFind:
    def __init__(self, elements):
        self.parent = {x: x for x in elements}
        self.rank = {x: 0 for x in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

def merge_lists(lists):
    # Flatten the list of lists and create a union-find structure
    elements = set(x for sublist in lists for x in sublist)
    uf = UnionFind(elements)

    # Union the elements in each sublist
    for sublist in lists:
        for i in range(1, len(sublist)):
            uf.union(sublist[0], sublist[i])

    # Collect the elements of each connected component
    components = {}
    for element in elements:
        root = uf.find(element)
        if root not in components:
            components[root] = []
        components[root].append(element)

    # Convert the components dictionary to a list of lists
    return list(components.values())

# Example usage
# lists = [[0, 3, 6], [1, 6, 7], [2, 5], [2, 9, 10], [3, 9]]
# merged_lists = merge_lists(lists)
# print(merged_lists)