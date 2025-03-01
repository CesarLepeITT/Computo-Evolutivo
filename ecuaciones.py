import math

class Ecuaciones:
    ############# Una dimensión ##############
    @staticmethod
    def F1(x):
        return abs(x[0]) + math.cos(x[0])

    @staticmethod
    def F2(x):
        return abs(x[0]) + math.sin(x[0])

    @staticmethod
    def F6(x):
        return (x[0] ** 2 + x[0]) * math.cos(x[0]) 

    ######## N dimensiones ###########
    @staticmethod
    def F3(x):
        return sum([x_i ** 2 for x_i in x])

    @staticmethod
    def F4(x):
        return sum(100 * ((x[n+1] - x[n]**2)**2) + ((1 - x[n])**2) for n in range(len(x) - 1))

    @staticmethod
    def F5(x):
        return sum(abs(x[n]) - 10 * math.cos(math.sqrt(abs(10 * x[n]))) for n in range(len(x)))

    @staticmethod    
    def F11(x):
        N = len(x)
        return 1 + sum((x[n] ** 2) / 4000 for n in range(N)) - math.prod(math.cos(x[n]) for n in range(N))

    ###### 2 Dimensiones #############
    @staticmethod
    def F7(x):
        return x[0] * math.sin(4 * x[0]) + 1.1 * x[1] * math.sin(2 * x[1])

    @staticmethod    
    def F12(x):
        return 0.5 + ((math.sin(math.sqrt(x[0]**2 + x[1]**2)))**2 - 0.5) / (1 + 0.1 * (x[0]**2 + x[1]))

    @staticmethod
    def F13(x):
        return (x[0]**2 + x[1]**2) ** 0.25 * math.sin(30 * ((x[0] + 0.5)**2 + x[1]**2)**0.1) + abs(x[0]) + abs(x[1])

    
    ### Grafica de curva de nivel para 1 y 2 dimensiones 