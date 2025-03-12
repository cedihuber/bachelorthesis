from setuptools import setup, find_packages

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    # This is the basic information about your project. Modify all this
    # information before releasing code publicly.
    name = 'attribute_cam',
    version = "0.1",
    description = 'Reproduce the paper "Visualization of Facial Attribute Classifiers via Class Activation Mapping"',

    url = 'https://gitlab.ifi.uzh.ch/aiml/projects/attribute-cam',
    license = 'BSD',
    author = 'Manuel Guenther',
    author_email = 'siebenkopf@googlemail.com',

    # If you have a better, long description of your package, place it on the
    # 'doc' directory and then hook it here
    long_description = open('README.md').read(),

    # This line is required for any distutils based packaging.
    # It will find all package-data inside the 'bob' directory.
    packages = find_packages('.'),
    include_package_data = True,

    # This line defines which packages should be installed when you "install"
    # this package. All packages that are mentioned here, but are not installed
    # on the current system will be installed locally and only visible to the
    # scripts of this package. Don't worry - You won't need administrative
    # privileges when using buildout.
    # install_requires = open("requirements.txt").read().split(),

    # This entry defines which scripts you will have inside the 'bin' directory
    # once you install the package (or run 'bin/buildout'). The order of each
    # entry under 'console_scripts' is like this:
    #   script-name-at-bin-directory = module.at.your.library:function
    #
    # The module.at.your.library is the python file within your library, using
    # the python syntax for directories (i.e., a '.' instead of '/' or '\').
    # This syntax also omits the '.py' extension of the filename. So, a file
    # installed under 'example/foo.py' that contains a function which
    # implements the 'main()' function of particular script you want to have
    # should be referred as 'example.foo:main'.
    #
    # In this simple example we will create a single program that will print
    # the version of bob.
    entry_points = {

      # scripts should be declared using this entry:
      'console_scripts' : [
          "extract_cams.py       = attribute_cam.script.extract_cams:main",
          "predict.py            = attribute_cam.script.predict:main",
          "average_cams.py       = attribute_cam.script.average_cams:main",
          "analyze.py            = attribute_cam.script.analyze:main",
          "error_rates.py        = attribute_cam.script.error_rates:main",
          "perturb_image.py        = attribute_cam.script.perturb_image:main"
      ],
    },

    # Classifiers are important if you plan to distribute this package through
    # PyPI. You can find the complete list of classifiers that are valid and
    # useful here (http://pypi.python.org/pypi?%3Aaction=list_classifiers).
    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
