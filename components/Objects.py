class Point:
    def __init__(self, xy, label=None, isIn=False, radius=2.5):
        self.xy = xy
        self.label = label
        self.isIn = isIn
        self.radius = radius


class Cluster:
    def __init__(self, xy, label, radius):
        self.xy = xy
        self.label = label
        self.radius = radius