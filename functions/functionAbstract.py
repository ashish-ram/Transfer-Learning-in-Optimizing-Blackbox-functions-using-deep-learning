from src import helper as h


class FunctionAbstract(object):
    def __init__(self, name=None, bounds=None, objective=None, pointsOfInterest=None, optimas=None):
        if name is not None:
            self.name = name
        self.bounds = bounds
        self.objective = objective

    def explore(self,resolution):
        if self.objective is None:
            raise NotImplementedError("This function object does not have an objective function")

        else:
            h.explore(self.objective,bounds=self.bounds, x1resolution=resolution, x2resolution=resolution, title=self.name)


