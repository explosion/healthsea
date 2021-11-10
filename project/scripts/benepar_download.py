import ssl
import benepar

"""This script uses a workaround for Mac users who might get problems with nltk.download()"""

try:
    benepar.download('benepar_en3')
except:
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    benepar.download('benepar_en3')