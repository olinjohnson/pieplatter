
class NotImplementedCorrectlyError(NotImplementedError):
    """Raised when a function or method has not been implemented correctly"""
    pass

class InvalidShapesError(Exception):
    """Raised when an argument passed to a function is not the expected shape"""
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
        raise NotImplementedCorrectlyError("Function ", func, "has not been implemented correctly")

    return throw_e

registry = []
def not_reviewed(func):

    def throw_e(*args, **kwargs):
        if func not in registry:
            print("\033[93m" + "WARNING: DANGEROUS FUNCTION ", func.__name__, func, "HAS NOT BEEN REVIEWED" + "\033[0m")
            registry.append(func)

        return func(*args, **kwargs)

    return throw_e
