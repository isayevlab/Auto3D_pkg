import warnings
from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from openeye import oechem
        from openeye import oequacpac
        from openeye import oeomega
    except:
        warnings.warn("Openeye software is not available.")

    try:
        import torchani
    except:
        warnings.warn("TorchANI is not installed")

    try:
        from Auto3D.batch_opt.ANI2xt_no_rep import ANI2xt
    except:
        warnings.warn("ANI2xt model is not available")
