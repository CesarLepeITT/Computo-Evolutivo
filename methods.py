import random
import math
from numba import njit
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
                n_runs -= 1
                list_convergence.append(func(best))
                if(n_runs <= 0):
                    break
                
            if (maximize and S_Quality > best_Quality) or (not maximize and S_Quality < best_Quality):
                best = S[:]
                best_Quality = S_Quality
            
            S = self.RandomSolution(n_dimentions)
            S_Quality = self.Quality(S, func)    
            
            
            
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
            exponent = (R_Quality - S_Quality if maximize else S_Quality - R_Quality) / temperature
            exponent = max(min(exponent, 700), -700)
            exponent = round(math.e**(exponent), 2)
            condition = random_number < exponent if maximize else random_number > exponent
            
            
            if (maximize and R_Quality > S_Quality) or (not maximize and R_Quality < S_Quality) or condition:
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
    
    def Perturb(self, S):
        return self.Tweak(S, self.maxDomainValue)
    
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
                list_convergence.append(func(best))
                if(n_runs <= 0):
                    break
                
            if (maximize and S_Quality > best_Quality) or (not maximize and S_Quality < best_Quality):
                best = S[:]
                best_Quality = S_Quality      
            
            H, H_Quality = NewHomeBase()
            S = self.Perturb(H)
            
        return best, list_convergence                     

    
    
    ##################################################33
    # METODO EVOLUTIVO
    ###################################################
    
    #BEGIN
    #   INITIALISE population with random candidate solutions;
    #   EVALUATE each candidate;
    #   REPEAT UNTIL ( TERMINATION CONDITION is satisfied ) DO
    #       SELECT parents;
    #       RECOMBINE pairs of parents;
    #       MUTATE the resulting offspring;
    #       EVALUATE new candidates;
    #       SELECT individuals for the next generation;
    #   OD
    # END

#• representation (definition of individuals)
#• evaluation function (or fitness function)
#• population
#• parent selection mechanism
#• variation operators, recombination and mutation
#• survivor selection mechanism (replacement)
#• termination condition

    @staticmethod
    def sort_key(individual):
        return individual[1]

    @staticmethod
    def initialize_population(num_individuals, num_genes, fitness_function):
        population = [[[random.random() for _ in range(num_genes)], None] for _ in range(num_individuals)]
        for i in range(num_individuals):
            population[i][1] = fitness_function(population[i][0])
        return population

    @staticmethod
    def select_parents(population, percentage_best, maximize):
        num_best = int(len(population) * percentage_best / 2)
        best_individuals = population[:num_best]
        worst_individuals = population[num_best:]
        random_individuals = []
        
        while len(random_individuals) + len(best_individuals) < int(len(population) / 2):
            rand_index = random.randint(0, len(worst_individuals) - 1)
            individual = worst_individuals.pop(rand_index)
            random_individuals.append(individual)
        
        parents = best_individuals + random_individuals
        
        parents = Methods.sort_population(parents, maximize)
        
        return parents
    @staticmethod
    def generate_offspring(old_population, parents, best_individual, population_size, fitness_function, min_value, max_value, mutation_prob, maximize):
        offspring = []
        
        while len(offspring) < population_size:
            operator_prob = random.uniform(0, 1)
            
            if operator_prob < 1 - mutation_prob:  # Crossover
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                children = Methods.recombine_individuals(parent1, parent2, min_value, max_value)
                
                child1 = children[0]
                child1[1] = fitness_function(child1[0])
                offspring.append(child1)
                
                if len(offspring) >= population_size:
                    break
                
                child2 = children[1]
                child2[1] = fitness_function(child2[0])
                offspring.append(child2)
            else:  # Mutation
                parent = random.choice(parents)
                mutated_individual = Methods.mutate_individual(parent, min_value, max_value)
                mutated_individual[1] = fitness_function(mutated_individual[0])
                
                offspring.append(mutated_individual)
        
        offspring = Methods.sort_population(offspring, maximize)

        if fitness_function(offspring[0][0]) > fitness_function(best_individual[0]):
            offspring[-1] = best_individual
            offspring = Methods.sort_population(offspring, maximize)
        
        old_population = old_population[:10]
        offspring = old_population + offspring[:population_size - 10]
        
        offspring = Methods.sort_population(offspring, maximize)

        # Elitismo: asegurar que el mejor individuo sobreviva
        if fitness_function(best_individual[0]) > fitness_function(offspring[0][0]):
            offspring[-1] = best_individual  # Reemplazar el peor
        else:
            offspring[0] = best_individual  # O ya era el mejor
        
        
        return offspring

    @staticmethod
    def mutate_individual(individual, min_value, max_value):
        gene_index = random.randint(0, len(individual[0]) - 1)
        mutated_individual = individual[:] 
        
        mutated_individual[0][gene_index] = random.uniform(min_value, max_value)
        print('mutacion')
        return mutated_individual

    @staticmethod
    def clamp_value(value, lower_bound, upper_bound):
        return max(lower_bound, min(value, upper_bound))

    @staticmethod
    def recombine_individuals(parent1, parent2, min_value, max_value):
        child1, child2 = [], []
        
        for gene_idx in range(len(parent1[0])):
            mendel_prob = random.uniform(0, 1)
            gene_child1 = parent1[0][gene_idx] * mendel_prob + (1 - mendel_prob) * parent2[0][gene_idx]
            gene_child2 = parent2[0][gene_idx] * mendel_prob + (1 - mendel_prob) * parent1[0][gene_idx]
            
            child1.append(Methods.clamp_value(gene_child1, min_value, max_value))
            child2.append(Methods.clamp_value(gene_child2, min_value, max_value))
        
        return [[child1, None], [child2, None]]

    @staticmethod
    def sort_population(population, maximize):
        population.sort(key=Methods.sort_key, reverse=maximize)

        return population

    def evolutionary_algorithm(self, fitness_function, num_individuals, num_genes, percentage_best, mutation_prob, num_iterations, maximize=True):
        population = Methods.initialize_population(num_individuals, num_genes, fitness_function)
        list_convergence = []

        population = Methods.sort_population(population, maximize)
        best_in_generation = population[0]
        
        list_convergence.extend([fitness_function(best_in_generation[0])] * num_individuals)
        num_iterations -= num_individuals
        
        while num_iterations > 0:
            parents = Methods.select_parents(population, percentage_best, maximize)
            test = population
            population = Methods.generate_offspring(population, parents, best_in_generation, num_individuals, fitness_function, self.minDomainValue, self.maxDomainValue, mutation_prob, maximize)
            population = Methods.sort_population(population, maximize)
            
            best_in_generation = population[0]
            
            print((min([x[1] for x in test]) >= best_in_generation[1]) == False)
            
            list_convergence.extend([best_in_generation[1]] * num_individuals)            
            num_iterations -= num_individuals
        print('-----------------------------')
        return population[0][0], list_convergence   

