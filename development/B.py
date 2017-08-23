import inspect
def f(ob):
    print(inspect.getmodule(inspect.stack()[1][0]))
    print(ob.__class__)
    print(inspect.getsourcelines(ob.__class__))