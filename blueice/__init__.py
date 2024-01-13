# Put core functionality at top level
# Modules below all have __all__ defined to make sure 'from ... import *' is well-behaved
# (well, exceptions does not, but it contains no shenanigans)
from .likelihood import *
from .model import *
from .source import *
from .exceptions import *
# from .inference import *   # Not needed, all the inference methods are added to LogLikelihood
from .parallel import *

__version__ = '1.2.0'
