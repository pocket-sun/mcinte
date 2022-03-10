import emcee

def read(filename):
    # retrieve data stored in #filename
    tmp = emcee.backends.HDFBackend(filename)
    return tmp
