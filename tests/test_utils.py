import os
import warnings
from Auto3D.auto3D import options
from Auto3D.utils import check_input, find_smiles_not_in_sdf


folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path1 = os.path.join(folder, "tests/files/all_stereo_centers_specified.smi")
path2 = os.path.join(folder, "tests/files/contain_unspecified_centers.smi")
path3 = os.path.join(folder, "tests/files/util_test.smi")
path4 = os.path.join(folder, "tests/files/util_test.sdf")


def test_check_input():
    """Test enumerate_isomer argument checker"""
    args1 = options(path1, k=1, enumerate_isomer=True, use_gpu=False)
    args1['input_format'] = 'smi'
    # count the number of warnings
    with warnings.catch_warnings(record=True) as warnings_list:
        check_input(args1)
    assert(len(warnings_list) == 0)

    args2 = options(path2, k=1, enumerate_isomer=True, use_gpu=False)
    args2['input_format'] = 'smi'
    # count the number of warnings
    with warnings.catch_warnings(record=True) as warnings_list2:
        check_input(args2)
    assert(len(warnings_list2) == 0)

    args3 = options(path2, k=1, use_gpu=False, enumerate_isomer=False)  # by default enumerate_isomer=True
    args3['input_format'] = 'smi'
    # count the number of warnings
    with warnings.catch_warnings(record=True) as warnings_list3:
        check_input(args3)
    assert(len(warnings_list3) >= 1)

    args4 = options(path1, k=1, use_gpu=False, enumerate_isomer=False)  # by default enumerate_isomer=True
    args4['input_format'] = 'smi'
    # count the number of warnings
    with warnings.catch_warnings(record=True) as warnings_list4:
        check_input(args4)
    print(warnings_list4, flush=True)
    assert(len(warnings_list4) == 0)


def test_find_smiles_not_in_sdf():
    """Test find_smiles_not_in_sdf"""
    bad = find_smiles_not_in_sdf(path3, path4)
    assert (len(bad) == 2)
    bad_ids = [id for id, _ in bad]
    assert ('9' in bad_ids)
    assert ('9b' in bad_ids)


# if __name__ == "__main__":
#     # test_check_input()
#     args4 = options(path1, k=1, use_gpu=False)  # by default enumerate_isomer=False
#     # count the number of warnings
#     check_input(args4)
#     with warnings.catch_warnings(record=True) as warnings_list4:
#         check_input(args4)
#     print(warnings_list4)
