class ProcInfo:
    def __init__(self, time, memory, points, arithmetic):
        self.time = time
        self.memory = memory
        self.points = points
        self.generations = len(points)
        self.arithmetic = arithmetic
