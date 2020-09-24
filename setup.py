import sys, os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
import subprocess
import platform

###########################
# SETUP BUILD ENVIRONMENT #
###########################
# CUDA support adapted from https://github.com/rmcgibbo/npcuda-example
def find_in_path(name, path):
    """Find a file in a search path
    adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    """
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDA_HOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be ' \
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig

def customize_compiler_for_nvcc(self, cudaconfig):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works to add compat. with cuda builds.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _compile methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', cudaconfig['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']


        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile

def install_setup_requirements(reqs):
    # install build dependencies
    subprocess.call(["pip", "install", *reqs])

#####################
# CUSTOMIZE BUILDER #
#####################
CUDA = locate_cuda()
def generate_custom_build_ext():
    # install build dependencies
    install_setup_requirements(['cython'])
    from Cython.Distutils import build_ext
    # run the customize_compiler by subclassing Cython's build_ext class and adding cuda pre-compilation
    class custom_build_ext(build_ext):
        def build_extensions(self):
            customize_compiler_for_nvcc(self.compiler, CUDA)
            build_ext.build_extensions(self)
    return custom_build_ext

def generate_cuda_extension():
    if platform.system() == 'Windows':
        raise Exception("Cuda Raytracing extensions on Windows are not supported. Falling back to CPU version")

    # install build dependencies
    install_setup_requirements(['numpy'])
    # Obtain the numpy include directory. This logic works across numpy versions.
    import numpy
    try:
        numpy_include = numpy.get_include()
    except AttributeError:
        numpy_include = numpy.get_numpy_include()

    # create a c++/CUDA extension module (library) to build during setup and include in package
    # find exact cuart.so to use - resolves conflict with cuda shared libs coming from other py packages
    cudart_filename = [x for x in os.listdir(CUDA['lib64']) if x.startswith('libcudart.so.')][0]
    ext = Extension(name='raytrace.raytrace_ext',
                    sources=['raytrace/raytrace_ext.pyx', pjoin('raytrace', 'src', 'raytrace.cu')],
                    library_dirs=[CUDA['lib64']],
                    libraries=[':'+cudart_filename],
                    language='c++',
                    runtime_library_dirs=[CUDA['lib64']],
                    # this syntax is specific to this build system
                    # we're only going to use certain compiler args with nvcc and not with gcc
                    # the implementation of this trick is in customize_compiler() below
                    extra_compile_args={'gcc': [],
                                        'nvcc': ['-arch=sm_30', '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'"]},
                    include_dirs = [numpy_include, CUDA['include'], pjoin('raytrace', 'src')])
    return ext

#############
# RUN SETUP #
#############
# run the customized setup
try:
    extensions = [generate_cuda_extension()]
except Exception as e:
    print(e)
    print('failed to build cuda extension.')
    extensions = []

setup(name='raytrace',
      author='Ryan Neph',
      author_email='neph320@gmail.com',
      version='1.4',

      packages=['raytrace',],
      ext_modules = extensions,
      install_requires = [
          'numpy',
      ],
      # inject our custom trigger
      cmdclass={
          'build_ext': generate_custom_build_ext()
      },
      # since the package has c code, the egg cannot be zipped
      zip_safe=False
      )

