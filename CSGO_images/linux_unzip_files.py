# https://github.com/ponty/pyunpack
"""
Installation on Ubuntu
$ sudo apt-get install unzip unrar p7zip-full
$ python3 -m pip install patool
$ python3 -m pip install pyunpack
"""
from pyunpack import Archive

Archive('CSGO_images.7z.001').extractall('.')
