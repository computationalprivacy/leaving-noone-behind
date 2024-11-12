from distutils.core import setup, Extension

module = Extension('cqbs', sources = ['optimqbs/qbs.c', 'optimqbs/cqbsmodule.c'])

setup (name = 'optimqbs',
	   version = '1.0',
	   description = 'C implementation of QBSes',
	   ext_modules = [module],
       packages = ['optimqbs'])