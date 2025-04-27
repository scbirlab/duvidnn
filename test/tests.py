import doctest
import duvida

if __name__ == '__main__':
    doctest.testmod(duvida.stateless.hessians)
    doctest.testmod(duvida.stateless.hvp)
    doctest.testmod(duvida.stateless.information)
