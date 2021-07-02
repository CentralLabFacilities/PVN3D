#!/usr/bin/env python3
import os
import sys
import site
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

working_dir = os.path.abspath(os.getcwd())
_ext_src_root = working_dir + "/pvn3d/_ext-src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))


setup(
    name='pvn3d',
	packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='pointnet2_utils._ext',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc": [
                    "-O2", "-I{}".format("{}/include".format(_ext_src_root))
                ],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


try:
    src_pth = './build'
    if os.environ.get('VIRTUAL_ENV'):
        tg_pth = os.environ.get('VIRTUAL_ENV') + '/lib/python{}.{}/site-packages'.format(sys.version_info[0],sys.version_info[1]) + '/pvn3d/lib/pointnet2_utils/'
    else:
        tg_pth = site.getusersitepackages() + '/pvn3d/lib/pointnet2_utils/'
    print("tg_pth", tg_pth)
    fd_lst = os.listdir(src_pth)
    for fd in fd_lst:
        if 'lib' in fd:
            src_pth = os.path.join(src_pth, fd, 'pointnet2_utils')
            f_nm = os.listdir(src_pth)[0]
            src_pth = os.path.join(src_pth, f_nm)
            tg_pth = os.path.join(tg_pth, f_nm)
    os.system('cp {} {}'.format(src_pth, tg_pth))
    print(
        src_pth, '==>', tg_pth,
    )
except:
    print(
        "\n****************************************************************\n",
        "Failed to copy builded .so to ./pvn3d/lib/pointnet2_utils/.\n",
        "Please manually copy the builded .so file (_ext.cpython*.so) in ./build"+\
        " to ./pvn3d/lib/pointnet2_utils/.",
        "\n****************************************************************\n"
    )

# vim: ts=4 sw=4 sts=4 expandtab
