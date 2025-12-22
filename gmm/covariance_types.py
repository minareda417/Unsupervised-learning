from enum import Enum

class CovarianceType(Enum):
    FULL = "full"
    TIED = "tied"
    DIAG = "diag"
    SPHERICAL = "spherical"
