import os
import warnings
from Auto3D.auto3D import options
from Auto3D.utils import check_input


folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path1 = os.path.join(folder, "tests/files/all_stereo_centers_specified.smi")
path2 = os.path.join(folder, "tests/files/contain_unspecified_centers.smi")


def test_check_input():
    """Test cis_trans argument checker"""
    args1 = options(path1, k=1, cis_trans=True, use_gpu=False)
    # count the number of warnings
    with warnings.catch_warnings(record=True) as warnings_list:
        check_input(args1)
    assert(len(warnings_list) == 0)

    args2 = options(path2, k=1, cis_trans=True, use_gpu=False)
    # count the number of warnings
    with warnings.catch_warnings(record=True) as warnings_list2:
        check_input(args2)
    assert(len(warnings_list2) == 0)

    args3 = options(path2, k=1, use_gpu=False)  # by default cis_trans=False
    # count the number of warnings
    with warnings.catch_warnings(record=True) as warnings_list3:
        check_input(args3)
    assert(len(warnings_list3) >= 1)

    args4 = options(path1, k=1, use_gpu=False)  # by default cis_trans=False
    # count the number of warnings
    with warnings.catch_warnings(record=True) as warnings_list4:
        check_input(args4)
    print(warnings_list4)
    assert(len(warnings_list4) == 0)


# if __name__ == "__main__":
#     # test_check_input()
#     args4 = options(path1, k=1, use_gpu=False)  # by default cis_trans=False
#     # count the number of warnings
#     check_input(args4)
#     with warnings.catch_warnings(record=True) as warnings_list4:
#         check_input(args4)
#     print(warnings_list4)
