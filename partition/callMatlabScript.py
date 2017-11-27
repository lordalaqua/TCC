import os


def callMatlabScript(script, *args, **kwargs):
    path = os.path.dirname(os.path.realpath(__file__))
    new_dir = os.path.join(path, 'matlab')
    arg_list = ' '.join(map(lambda arg: str(arg), args))
    verbose = ' > nul 2>&1'
    return not os.system('octave-cli --eval "cd %s; %s %s"%s' %
                         (new_dir, script, arg_list, verbose))
