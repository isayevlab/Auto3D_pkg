import os
from send2trash import send2trash
from Auto3D.auto3D import options, main


folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(folder, "tests/files/smiles2.smi")


def test_auto3D():
    """Check that the program runs"""
    args = options(path, k=1, use_gpu=False, convergence_threshold=0.01)
    out = main(args)
    out_folder = os.path.dirname(os.path.abspath(out))
    send2trash(out_folder)
