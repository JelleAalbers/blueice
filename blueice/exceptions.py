class NoOpimizationNecessary(Exception):
    pass


class OptimizationFailed(Exception):
    pass


class NotPreparedException(Exception):
    pass


class NoShapeParameters(Exception):
    pass


class InvalidParameter(Exception):
    """A particular parameter to the likelihood is not present"""
    pass


class InvalidParameterSpecification(Exception):
    """An add_x_parameter method was called wrongly"""
    pass
