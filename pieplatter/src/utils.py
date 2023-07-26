
class NotImplementedCorrectlyError(NotImplementedError):
    """Raised when a function or method has not been implemented correctly"""
    pass

class InvalidShapesError(Exception):
    pass

class ParametersNotDefinedError(Exception):
    """Raised when the network parameters have not been defined appropriately"""
    pass

def not_implemented(func):

    def throw_e(*args, **kwargs):
        raise NotImplementedError()

    return throw_e

def not_implemented_correctly(func):

    def throw_e(*args, **kwargs):
        raise NotImplementedCorrectlyError()

    return throw_e

def not_reviewed(func):

    def throw_e(*args, **kwargs):
        print("\033[93m" + "WARNING: DANGEROUS FUNCTION ", func, " HAS NOT BEEN REVIEWED" + "\033[0m")
        return func(*args, **kwargs)

    return throw_e
