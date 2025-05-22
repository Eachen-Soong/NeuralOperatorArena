# from .fno import TFNO, TFNO1d, TFNO2d, TFNO3d
# from .fno import FNO, FNO1d, FNO2d, FNO3d
from .fno import SFNO
from .uno import UNO
from .new_fno import F_FNO2D, FNO, SpecProdFNO, DimFNO
from .FNO_Original import FNO_2D as FNO_2D_Original
from .FNO_Original import ProdFNO_2D_test as ProdFNO_2D_Original
from .FNO_Original import FNO_1D as FNO_1D_Original
from .FNO_Original import ProdFNO_1D as ProdFNO_1D_Original

# from .prod_fno import ProdFNO
# from .fnogno import FNOGNO
from .model_dispatcher import get_model

from .LSM.LSM_2D import LSM_2D
from .FNO_Original import FNO_2D_test1