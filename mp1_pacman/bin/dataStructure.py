class Queue:
    def __init__(self):
        raise NotImplementedError
    def extend(self, items):
        for item in items:
            self.append(item)



def Stack():
    'list is FILO structure'
    return []

class FIFOQueue(Queue):
    
    def __init__(self):
        self.A = []

    def append(self,item):
        self.A.append(item)

    def __len__(self):
        return len(self.A)

    def extend(self,items):
        for item in items:
            self.A.append(item)
        #print '********************\n',self.A

    def pop(self):
        return self.A.pop(0) 

    def __contains__(self,item):
        return item in self.A


import bisect

class PriorityQueue(Queue):
    
    'the left column is f(x), the right one is x'
    def __init__(self, order=min, f=lambda x:x):
        self.A = []
        self.order = order
        self.f = f

    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.A)

    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item
    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)
    def clear(self):
        while len(self):
            self.pop()
