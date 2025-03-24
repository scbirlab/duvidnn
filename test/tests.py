import doctest
import duvida as dv

if __name__ == '__main__':
    doctest.testmod(dv.stateless.hessians)
    doctest.testmod(dv.stateless.hvp)
    doctest.testmod(dv.stateless.information)