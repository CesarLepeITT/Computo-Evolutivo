import random
import math

class Methods:
    def __init__(self, minDomainValue, maxDomainValue):
        self.maxDomainValue = maxDomainValue
        self.minDomainValue = minDomainValue

    def Quality(self, S, func):
        return func(S)

    def Tweak(self, S, maxStep):        
        def TweakValue(value):
            prob = random.uniform(0, 1)
            number = random.uniform(0, maxStep)
            
            if prob > 2/3:
                value += number 
            elif prob > 1/3 and prob < 2/3:
                value -= number

            if value > self.maxDomainValue:
                value = self.maxDomainValue
            elif value < self.minDomainValue:
                value = self.minDomainValue
            
            return value

        return [TweakValue(value) for value in S]
        
    def RandomSolution(self, n_numeros):
        return [random.uniform(self.minDomainValue, self.maxDomainValue) for _ in range(n_numeros)]

    def HillClimbing(self, func, maxStep, n_runs, n_dimentions, maximize=True):
        list_convergence = []
        
        S = self.RandomSolution(n_dimentions)
        S_Quality = self.Quality(S, func)

        for i in range(n_runs):
            R = self.Tweak(S, maxStep)
            R_Quality = self.Quality(R, func)
            
            if (maximize and R_Quality > S_Quality) or (not maximize and R_Quality < S_Quality):
                S = R
                S_Quality = R_Quality
            
            list_convergence.append(func(S))                
                
                
        return S, list_convergence
    
    def SteepestAscentHillClimbing(self, func, maxStep, n_runs, n_dimentions, n_tweaks, maximize=True):
        list_convergence = []
        
        
        S = self.RandomSolution(n_dimentions)
        S_Quality = self.Quality(S, func)

        for i in range(n_runs):
            R = self.Tweak(S, maxStep)
            R_Quality = self.Quality(R, func)
            
            for j in range(n_tweaks):
                W = self.Tweak(S, maxStep)
                W_Quality = self.Quality(W, func)
                if (maximize and W_Quality > R_Quality) or (not maximize and W_Quality < R_Quality):
                    R = W
                    R_Quality = W_Quality

            if (maximize and R_Quality > S_Quality) or (not maximize and R_Quality < S_Quality):
                S = R
                S_Quality = R_Quality
                
            list_convergence.append(func(S))
        
        return S, list_convergence
    
    def SteepestAscentHillClimbingWithReplacement(self, func, maxStep, n_runs, n_dimentions, n_tweaks, maximize=True):
        list_convergence = []
        
        S = self.RandomSolution(n_dimentions)
        S_Quality = self.Quality(S, func)

        best = S[:]
        best_Quality = S_Quality

        for i in range(n_runs):
            R = self.Tweak(S, maxStep)
            R_Quality = self.Quality(R, func)
            
            for j in range(n_tweaks):
                W = self.Tweak(S, maxStep)
                W_Quality = self.Quality(W, func)
                if (maximize and W_Quality > R_Quality) or (not maximize and W_Quality < R_Quality):
                    R = W
                    R_Quality = W_Quality

            S = R[:]
            S_Quality = R_Quality
            
            if (maximize and S_Quality > best_Quality) or (not maximize and S_Quality < best_Quality):
                best = S[:]
                best_Quality = S_Quality
        
            list_convergence.append(func(best))
        
        return best, list_convergence
    
    def RandomSearch(self, func, n_runs, n_dimentions, maximize=True):
        list_convergence = []
        
        best = self.RandomSolution(n_dimentions)
        best_Quality = self.Quality(best, func)
        
        for i in range(n_runs):
            S = self.RandomSolution(n_dimentions)
            S_Quality = self.Quality(S, func)
            if (maximize and S_Quality > best_Quality) or (not maximize and S_Quality < best_Quality):
                best = S[:]
                best_Quality = S_Quality
            list_convergence.append(func(best))
        return best, list_convergence

######################### corregir xddd

    def RandomInterval(self, interval, nNumbers):
        return [random.randint(interval[0], interval[1]) for _ in range(nNumbers)]

        
    def HillClimbingWithRandomRestarts(self, func, maxStep, n_runs, n_dimentions, intervals ,maximize=True):
        list_convergence = []
        T = self.RandomInterval(intervals, 100)
        
        S = self.RandomSolution(n_dimentions)
        S_Quality = self.Quality(S, func)

        best = S[:]
        best_Quality = S_Quality
        
        while(n_runs > 0):
            time = T[random.randint(0,100 - 1)]
            
            while(time > 0):
                R = self.Tweak(S, maxStep)
                R_Quality = self.Quality(R, func)
                
                if (maximize and R_Quality > S_Quality) or (not maximize and R_Quality < S_Quality):
                    S = R[:]
                    S_Quality = R_Quality
                    
                time -= 1
                n_runs -=1
            
            if (maximize and S_Quality > best_Quality) or (not maximize and S_Quality < best_Quality):
                best = S[:]
                best_Quality = S_Quality
            
            S = self.RandomSolution(n_dimentions)
            S_Quality = self.Quality(S, func)    
            
            list_convergence.append(func(best))
            
        return best, list_convergence    

    def SimmulatedAnneling(self, func, maxStep, n_runs, n_dimentions, temperature, temperatureDecrease, maximize=True):
        list_convergence = []
        S = self.RandomSolution(n_dimentions)
        S_Quality = self.Quality(S, func)
        
        best = S[:]
        best_Quality = S_Quality
        
        for i in range(n_runs):
            R = self.Tweak(S, maxStep)
            R_Quality = self.Quality(R, func)
            
            random_number = random.randint(0, 100) / 100
            
            if (maximize and R_Quality > S_Quality) or (not maximize and R_Quality < S_Quality) or random_number < math.e**((R_Quality - S_Quality) / temperature):
                S = R[:]
                S_Quality = R_Quality
                
            temperature -= temperatureDecrease
            
            if(temperature == 0):
                temperature = 1
            
            
            
            if (maximize and S_Quality > best_Quality) or (not maximize and S_Quality < best_Quality):
                best = S[:]
                best_Quality = S_Quality            
                
            list_convergence.append(func(best))    
        
        return best,list_convergence
    
    def Perturb(self, S, maxStep):
        return self.Tweak(S, maxStep * 10)
    
    def IteratedLocalSearchWithRandomRestarts(self, func, n_runs, n_dimentions, maxStep, intervals, maximize = True):
        def NewHomeBase():
            if(maximize and S_Quality > H_Quality) or (not maximize and S_Quality < H_Quality):
                return S, S_Quality        
            return H, H_Quality 
        
        list_convergence=[]
        T = self.RandomInterval(intervals, 100)
        
        S = self.RandomSolution(n_dimentions)
        S_Quality = self.Quality(S, func)
        
        H = S
        H_Quality= S_Quality
        
        best = S
        best_Quality = S_Quality
        
        
        while(n_runs > 0):
            time = T[random.randint(0,100 - 1)]
            
            while(time > 0):
                R = self.Tweak(S, maxStep)
                R_Quality = self.Quality(R, func)
                
                if (maximize and R_Quality > S_Quality) or (not maximize and R_Quality < S_Quality):
                    S = R[:]
                    S_Quality = R_Quality
                    
                time -= 1
                n_runs -= 1
            
            if (maximize and S_Quality > best_Quality) or (not maximize and S_Quality < best_Quality):
                best = S[:]
                best_Quality = S_Quality      
            
            H, H_Quality = NewHomeBase()
            S = self.Perturb(H, maxStep)
            list_convergence.append(func(best))
            
        return best, list_convergence                     
        
        
