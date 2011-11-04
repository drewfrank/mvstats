#!/usr/bin/env python
# Do not add setuptools here; use setupegg.py instead. Nose still has problems running
# tests inside of egg packages, so it is useful to be able to install without eggs as needed.
from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info
import os, sys
#try:
#    from googlecode import upload2google
#except:
#    def upload2google():
#        raise "This commmand is only available with the subversion development version."

config = Configuration('mvstats',parent_package=None,top_path=None)
dist = sys.argv[1]


# ==============================
# = Compile Fortran extensions =
# ==============================

# If optimized lapack/ BLAS libraries are present, compile distributions that involve linear algebra against those.
# Otherwise compile blas and lapack from netlib sources.
lapack_info = get_info('lapack_opt',1)
f_sources = ['mvstats/flib.f', 'mvstats/flib_blas.f', 'mvstats/blas_wrap.f'
    , 'cephes/i0.c', 'cephes/c2f.c','cephes/chbevl.c']
if lapack_info:
    config.add_extension(name='flib',sources=f_sources, extra_info=lapack_info, f2py_options=['skip:ppnd7'])

if not lapack_info or dist in ['bdist', 'sdist']:
    ##inc_dirs = ['blas/BLAS','lapack/double']
    print 'No optimized BLAS or Lapack libraries found, building from source. This may take a while...'
    for fname in os.listdir('blas/BLAS'):
        # Make sure this is a Fortran file, and not one of those weird hidden files that
        # pop up sometimes in the tarballs
        if fname[-2:]=='.f' and fname[0].find('_')==-1:
            f_sources.append('blas/BLAS/'+fname)
    ##    for fname in os.listdir('lapack/double'):
    ##        if fname[-2:]=='.f':
    ##            inc_dirs.append('lapack/double/'+fname)

    for fname in ['dpotrs','dpotrf','dpotf2','ilaenv','dlamch','ilaver','ieeeck','iparmq']:
        f_sources.append('lapack/double/'+fname+'.f')
    config.add_extension(name='flib',sources=f_sources)


config_dict = config.todict()
try:
    config_dict.pop('packages')
except:
    pass


# ===========================================
# = Compile GP package's Fortran extensions =
# ===========================================

# # Compile linear algebra utilities
# if lapack_info:
    # config.add_extension(name='gp.linalg_utils',sources=['mvstats/gp/linalg_utils.f','mvstats/blas_wrap.f'], extra_info=lapack_info)
    # config.add_extension(name='gp.incomplete_chol',sources=['mvstats/gp/incomplete_chol.f'], extra_info=lapack_info)

# if not lapack_info or dist in ['bdist', 'sdist']:
    # print 'No optimized BLAS or Lapack libraries found, building from source. This may take a while...'
    # f_sources = ['mvstats/blas_wrap.f']
    # for fname in os.listdir('blas/BLAS'):
        # if fname[-2:]=='.f':
            # f_sources.append('blas/BLAS/'+fname)

    # for fname in ['dpotrs','dpotrf','dpotf2','ilaenv','dlamch','ilaver','ieeeck','iparmq']:
        # f_sources.append('lapack/double/'+fname+'.f')

    # config.add_extension(name='gp.linalg_utils',sources=['mvstats/gp/linalg_utils.f'] + f_sources)
    # config.add_extension(name='gp.incomplete_chol',sources=['mvstats/gp/incomplete_chol.f'] + f_sources)


# #Compile covariance functions
# config.add_extension(name='gp.cov_funs.isotropic_cov_funs',\
# sources=['mvstats/gp/cov_funs/isotropic_cov_funs.f','blas/BLAS/dscal.f'],\
# extra_info=lapack_info)

# config.add_extension(name='gp.cov_funs.distances',sources=['mvstats/gp/cov_funs/distances.f'], extra_info=lapack_info)




if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(  version="0.1",
            description="Some common multivariate distributions",
            #maintainer="David Huard",
            #maintainer_email="david.huard@gmail.com",
            author="Levi Boyles, Drew Frank",
            author_email="drewfrank@gmail.com",
            url="#",
            #download_url="",
            license="MIT",
            classifiers=[
                'Development Status :: 1 - Alpha',
                'Environment :: Console',
                'Operating System :: OS Independent',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: MIT',
                'Programming Language :: Python',
                'Programming Language :: Fortran',
                'Topic :: Scientific/Engineering',
                 ],
            requires=['NumPy (>=1.3)',],
            long_description="""
            This package fills a gap in scipy.stats by providing PDFs and sampling 
            functions for some common multivariate distributions.  The code is 
            heavily based on PyMC's distributions.py and associated Fortran libraries.

            ``mvstats`` only requires ``NumPy``.
            """,
            packages=["mvstats"],
            #cmdclass={'upload2google':upload2google},
            **(config_dict))
