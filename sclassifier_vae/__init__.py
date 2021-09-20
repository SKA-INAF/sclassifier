#! /usr/bin/env python

__title__ = 'sclassifier_vae'
__version__ = '1.0.0'
__author__ = 'Simone Riggi'
__license__ = 'GPL3'
__date__ = '2019-05-29'
__copyright__ = 'Copyright 2019 by Simone Riggi - INAF'


import logging
import logging.config


# Create the Logger
#logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
logging.basicConfig(format="%(asctime)-15s %(levelname)s - %(message)s",datefmt='%Y-%m-%d %H:%M:%S')
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("This is sclassifier_vae {0}-({1})".format(__version__, __date__))


