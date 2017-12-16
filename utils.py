
import functools
import operator
import time
import sys
import click
import os.path


def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def product(iterable):
    return functools.reduce(operator.mul, iterable, 1)


class Timer:
    def __init__(self, name=None, verbose=True):
        if verbose:
            assert name is not None

        self.name = name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            sys.stdout.write(self.name + '... ')
            sys.stdout.flush()

        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.duration = self.end - self.start

        if self.verbose:
            duration_string = '{0:.2f}s'.format(self.duration)
            sys.stdout.write(duration_string + '\n')
            sys.stdout.flush()

    def get_duration(self):
        return self.duration



class Params:
    """
    Simple wrapper class for dictionaries to make each key accessible as an attribute.

    >>> p = Params({'a': 1, 'b': 2})
    >>> print(p.a, p.b)
    ... 1 2
    """

    def __init__(self, params_dict):
            self.__dict__ = params_dict

    def __repr__(self):
        classname = self.__class__.__name__
        return '{}[{}]'.format(classname, self.__dict__)

    def pretty_print(self):
        extra_space = 2

        keys = self.__dict__.keys()
        max_length = max(len(str(key)) for key in keys)

        for key in sorted(keys):
            row = ''
            row += str(key)
            row += ' ' * ((max_length - len(key)) + extra_space)
            value = self.__dict__[key]
            row += '{} [{}]'.format(value, type(value).__name__)
            print(row)


def check(fn, error_message=None):
    """
    Creates callback function which raises click.BadParameter when `fn` returns `False` on given input.

    >>> @click.command()
    >>> @click.option('--probability', callback=check(lambda x: 0 <= x <= 1, "--probability must be between 0 and 1 (inclusive)"))
    >>> def f(probability):
    >>>     print('P:', probability)
    """
    def f(ctx, param, value):
        if fn(value):
            return value
        else:
            if error_message is None:
                msg = str(value)
            else:
                msg = '{}. Value: {}'.format(error_message, value)

            raise click.BadParameter(msg)

    return f


def listdir(path, extensions=None):
    return sorted([f for f in os.listdir(path) if not f.startswith('.') and f.lower().endswith(extensions)])
