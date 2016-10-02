# Put core functionality at top level
# Modules below all have __all__ defined to make sure 'from ... import *' is well-behaved
from .likelihood import *
from .model import *
from .source import *
# from .inference import *   # Not needed, all the inference methods are added to LogLikelihood
from .parallel import *

__version__ = '1.0.0'
