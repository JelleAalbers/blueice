class BlueIceException(Exception):
    pass


class NoOpimizationNecessary(BlueIceException):
    pass


class OptimizationFailed(BlueIceException):
    pass


class NotPreparedException(BlueIceException):
    pass


class NoShapeParameters(BlueIceException):
    pass


class InvalidParameter(BlueIceException):
    """A particular parameter to the likelihood is not present"""
    pass


class InvalidParameterSpecification(BlueIceException):
    """An add_x_parameter method was called wrongly"""
    pass


class PDFNotComputedException(BlueIceException):
    pass
