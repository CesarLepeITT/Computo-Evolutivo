from methods import Methods as m
from ecuaciones import Ecuaciones as eq
import os
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import random
from numba import njit

class Experiment:
    @staticmethod            
    def evaluate(func, maxStep, n_experiments, n_runs, n_dimentions, n_tweaks, interval, minDomainValue, maxDomainValue, temperature, temperatureDecrease, maximize, num_individuals, percentage_best, mutation_prob):
        def add_list(l1, l2):
            l2 = [x/ n_experiments for x in l2]
            return [a + b for a, b in zip(l1, l2)]
    
        # Crear un DataFrame vacío para almacenar los resultados
        stats = pd.DataFrame(columns=[
            'HillClimbing', 'SteepestAscentHillClimbing', 'SteepestAscentHillClimbingWithReplacement',
            'RandomSearch', 'HillClimbingWithRandomRestarts', 'SimmulatedAnneling', 'IteratedLocalSearchWithRandomRestarts', 'Evolutivo'
        ])    

        hill_climbing_dict = {'Hill Climbing': [0] * n_runs}
        steepest_ascent_dict = {'Steepest Ascent': [0] * n_runs}
        steepest_ascent_with_replacement_dict = {'Steepest Ascent With Replacement': [0] * n_runs}
        random_search_dict = {'Random Search': [0] * n_runs}
        hill_climbing_with_restarts_dict = {'Hill Climbing With Restarts': [0] * n_runs}
        simulated_annealing_dict = {'Simulated Annealing': [0] * n_runs}
        iterated_local_search_dict = {'Iterated Local Search': [0] * n_runs}
        evolutivo_dict = {'Evolutivo': [0] * n_runs}
        # Inicializar el método
        method = m(minDomainValue, maxDomainValue)
        # Iterar sobre el número de experimentos
        for exp in range(n_experiments):
            # Crear una lista para almacenar los resultados de cada método
            experiment_results = []

            # Llamar a cada función y agregar su resultado a la lista
            hill_climbing_result = method.HillClimbing(func=func, maxStep=maxStep, n_runs=n_runs, n_dimentions=n_dimentions, maximize=maximize)
            experiment_results.append(func(hill_climbing_result[0]))
            hill_climbing_dict['Hill Climbing'] = add_list(hill_climbing_dict['Hill Climbing'], hill_climbing_result[1])
            #hill_climbing_dict['Hill Climbing'] = [x/n_experiments for x in hill_climbing_dict['Hill Climbing']]

            steepest_ascent_result = method.SteepestAscentHillClimbing(func=func, maxStep=maxStep, n_runs=n_runs, n_tweaks=n_tweaks, maximize=maximize, n_dimentions=n_dimentions)
            experiment_results.append(func(steepest_ascent_result[0]))
            steepest_ascent_dict['Steepest Ascent'] = add_list(steepest_ascent_dict['Steepest Ascent'], steepest_ascent_result[1])
            #steepest_ascent_dict['Steepest Ascent'] = [x/n_experiments for x in steepest_ascent_dict['Steepest Ascent']]
            
            steepest_ascent_with_replacement_result = method.SteepestAscentHillClimbingWithReplacement(func=func, maxStep=maxStep, n_runs=n_runs, n_dimentions=n_dimentions, n_tweaks=n_tweaks, maximize=maximize)
            experiment_results.append(func(steepest_ascent_with_replacement_result[0]))
            steepest_ascent_with_replacement_dict['Steepest Ascent With Replacement'] = add_list(steepest_ascent_with_replacement_dict['Steepest Ascent With Replacement'], steepest_ascent_with_replacement_result[1])
            #steepest_ascent_with_replacement_dict['Steepest Ascent With Replacement'] = [x/n_experiments for x in steepest_ascent_with_replacement_dict['Steepest Ascent With Replacement']]
            
            random_search_result = method.RandomSearch(func=func, n_runs=n_runs, n_dimentions=n_dimentions, maximize=maximize)
            experiment_results.append(func(random_search_result[0]))
            random_search_dict['Random Search'] = add_list(random_search_dict['Random Search'], random_search_result[1])
            #random_search_dict['Random Search'] = [x/n_experiments for x in random_search_dict['Random Search']]
            
            hill_climbing_with_restarts_result = method.HillClimbingWithRandomRestarts(func=func, maxStep=maxStep, n_runs=n_runs, n_dimentions=n_dimentions, intervals=interval, maximize=maximize)
            experiment_results.append(func(hill_climbing_with_restarts_result[0]))
            hill_climbing_with_restarts_dict['Hill Climbing With Restarts'] = add_list(hill_climbing_with_restarts_dict['Hill Climbing With Restarts'], hill_climbing_with_restarts_result[1])
            #hill_climbing_with_restarts_dict['Hill Climbing With Restarts'] = [x/n_experiments for x in hill_climbing_with_restarts_dict['Hill Climbing With Restarts']]
            
            simulated_annealing_result = method.SimmulatedAnneling(func=func, maxStep=maxStep, n_runs=n_runs, n_dimentions=n_dimentions, temperature=temperature, temperatureDecrease=temperatureDecrease, maximize=maximize)
            experiment_results.append(func(simulated_annealing_result[0]))
            simulated_annealing_dict['Simulated Annealing'] = add_list(simulated_annealing_dict['Simulated Annealing'], simulated_annealing_result[1])
            #simulated_annealing_dict['Simulated Annealing'] = [x/n_experiments for x in simulated_annealing_dict['Simulated Annealing']]
            
            
            iterated_local_search_result = method.IteratedLocalSearchWithRandomRestarts(func=func, n_runs=n_runs, n_dimentions=n_dimentions, maxStep=maxStep, intervals=interval, maximize=maximize)
            experiment_results.append(func(iterated_local_search_result[0]))
            iterated_local_search_dict['Iterated Local Search'] = add_list(iterated_local_search_dict['Iterated Local Search'], iterated_local_search_result[1])
            #iterated_local_search_dict['Iterated Local Search'] = [x/n_experiments for x in iterated_local_search_dict['Iterated Local Search']]
            
            evolutivo_result = method.evolutionary_algorithm(func, num_individuals, n_dimentions, percentage_best, mutation_prob, n_runs, maximize)
            experiment_results.append(func(evolutivo_result[0]))
            evolutivo_dict['Evolutivo'] = add_list(evolutivo_dict['Evolutivo'], evolutivo_result[1])
            #evolutivo_dict['Evolutivo'] = [x/n_experiments for x in evolutivo_dict['Evolutivo']]
        
            stats.loc[exp] = experiment_results
        
        # Lista de diccionarios con los resultados acumulados
        lista = [
            hill_climbing_dict,
            steepest_ascent_dict,
            steepest_ascent_with_replacement_dict,
            random_search_dict,
            hill_climbing_with_restarts_dict,
            simulated_annealing_dict,
            iterated_local_search_dict,
            evolutivo_dict
        ]
        stats.to_csv('stats.csv', mode='a')
        
        return stats, lista 

    @staticmethod
    def to_table(data:pd.DataFrame, func:eq, maximize:bool):
        def IQR(column):
            return column.quantile(0.75) - column.quantile(0.25)  

        tabla = pd.DataFrame(columns=['Problema', 'Método', 'Máximo', 'Mínimo', 'Mediana', 'IQR', 'Media', 'STD', 'Mejor Solución'])

        for i in range(len(data.columns)):
            
            column = data.iloc[:,i]
            
            row = [
                func.__name__,   
                data.columns[i],      
                column.max(),    
                column.min(),    
                column.median(), 
                IQR(column),     
                column.mean(),   
                column.std(),    
                column.max() if maximize else column.min()  
            ]
            
            tabla.loc[i] = row 
                    
        return tabla

    @staticmethod
    def concat_stats(parameters_list, funciones, titulo_grafica):
        df_list = []  # Lista para acumular DataFrames
        lista_convergencias = [] 
        
        for parameters in parameters_list: # Esto recorre cada ecuacion 
            stats, dict_convergencia = Experiment.evaluate(**parameters)
            lista_convergencias.append(dict_convergencia)
            df_list.append(Experiment.to_table(
                stats,
                parameters['func'],
                parameters['maximize']
            ))

        Experiment.convergencia(lista_convergencias, funciones, titulo_grafica)

        return pd.concat(df_list, ignore_index=True)  
        
    @staticmethod        
    def experiment1dim(n_experiments:int, n_runs:int, titulo_grafica):
        ############# Una dimensión ##############
        funciones = [eq.F1, eq.F2, eq.F6]
        n_tweaks = int(n_runs/4)
        interval = [int(n_runs/5),int(n_runs/2)]
        
        num_individuals = 50
        percentage_best = 0.8
        mutation_prob = 0.2
        
        parameters_list = [{
            'func' : eq.F1,
            'maxStep' : 20,
            'n_experiments': n_experiments,
            'n_runs' : n_runs, 
            'n_dimentions' : 1, 
            'n_tweaks' : n_tweaks, 
            'interval' : interval, 
            'minDomainValue' : -20,
            'maxDomainValue' : 20, 
            'temperature' : 1000, 
            'temperatureDecrease' : 100, 
            'maximize' : False,
            'num_individuals': 300,
            'percentage_best': 0.9,
            'mutation_prob': 0
            },{
            'func' : eq.F2,
            'maxStep' : 20,
            'n_experiments': n_experiments,
            'n_runs' : n_runs, 
            'n_dimentions' : 1, 
            'n_tweaks' : n_tweaks, 
            'interval' : interval, 
            'minDomainValue' : -20,
            'maxDomainValue' : 20, 
            'temperature' : 1000, 
            'temperatureDecrease' : 100, 
            'maximize' : False,
            'num_individuals': 200,
            'percentage_best': 0.9,
            'mutation_prob': 0.3
            },{
            'func' : eq.F6,
            'maxStep' : 20,
            'n_experiments': n_experiments,
            'n_runs' : n_runs, 
            'n_dimentions' : 1, 
            'n_tweaks' : n_tweaks, 
            'interval' : interval, 
            'maxDomainValue' : 10,
            'minDomainValue' : -10,
            'temperature' : 1000, 
            'temperatureDecrease' : 100, 
            'maximize' : False ,
            'num_individuals': num_individuals,
            'percentage_best': percentage_best,
            'mutation_prob': mutation_prob
            }]
        
        stats = Experiment.concat_stats(parameters_list, funciones, titulo_grafica)
        nombre = 'UnaDimension'
        stats.to_csv(f'csv/{nombre}.csv',index=False)
        stats.to_latex(f'tex/{nombre}.tex', index=False)
        
        return stats
                   
    @staticmethod         
    def experiment2dim(n_experiments:int, n_runs:int,titulo_grafica):    
        ###### 2 Dimensiones #############
        funciones = [eq.F7, eq.F12, eq.F13]
        n_tweaks = int(n_runs/4)
        interval = [int(n_runs/5),int(n_runs/2)]
        
        num_individuals = 10
        percentage_best = 1
        mutation_prob = 0.1
        parameters_list = [{
            'func' : eq.F7,
            'maxStep' : 4,
            'n_experiments': n_experiments,
            'n_runs' : n_runs, 
            'n_dimentions' : 2, 
            'n_tweaks' : n_tweaks, 
            'interval' : interval, 
            'maxDomainValue' : 10, 
            'minDomainValue' : 0, 
            'temperature' : 1000, 
            'temperatureDecrease' : 100, 
            'maximize' : False ,
            'num_individuals': 50,
            'percentage_best': 0.9,
            'mutation_prob': 0.1
            },{
            'func' : eq.F12,
            'maxStep' : 4,
            'n_experiments': n_experiments,
            'n_runs' : n_runs, 
            'n_dimentions' : 2, 
            'n_tweaks' : n_tweaks, 
            'interval' : interval, 
            'maxDomainValue' : 5, 
            'minDomainValue' : -5, 
            'temperature' : 1000, 
            'temperatureDecrease' : 100, 
            'maximize' : False ,
            'num_individuals': num_individuals,
            'percentage_best': percentage_best,
            'mutation_prob': mutation_prob
            },{
            'func' : eq.F13,
            'maxStep' : 5,
            'n_experiments': n_experiments,
            'n_runs' : n_runs, 
            'n_dimentions' : 2, 
            'n_tweaks' : n_tweaks, 
            'interval' : interval, 
            'maxDomainValue' : 10, 
            'minDomainValue' : -10,
            'temperature' : 1000, 
            'temperatureDecrease' : 100, 
            'maximize' : False ,
            'num_individuals': 100,
            'percentage_best': 0.3,
            'mutation_prob': 0.1
            }]     
        
        stats = Experiment.concat_stats(parameters_list, funciones,titulo_grafica)
        nombre = 'DosDimensiones'
        stats.to_csv(f'csv/{nombre}.csv',index=False)
        stats.to_latex(f'tex/{nombre}.tex', index=False)
        
        return stats
    
    @staticmethod
    def experimentNdim(n_dim:int, n_experiments:int, n_runs:int,titulo_grafica, configuracions):  
        ######## N dimensiones ###########
        funciones = [eq.F3, eq.F4, eq.F5, eq.F11]
        
        n_tweaks = int(n_runs/7)
        interval = [int(n_runs/10),int(n_runs/5)]
        
        parameters_list = [{
            'func' : eq.F3,
            'maxStep' : 4,
            'n_experiments': n_experiments,
            'n_runs' : n_runs, 
            'n_dimentions' : n_dim, 
            'n_tweaks' : n_tweaks, 
            'interval' : interval, 
            'maxDomainValue' : 5, 
            'minDomainValue' : -5, 
            'temperature' : 1000, 
            'temperatureDecrease' : 100, 
            'maximize' : False ,
            'num_individuals': configuracions[0][0],
            'percentage_best': configuracions[0][1],
            'mutation_prob': configuracions[0][2]
            },{
            'func' : eq.F4,
            'maxStep' : 0.4,
            'n_experiments': n_experiments,
            'n_runs' : n_runs, 
            'n_dimentions' : n_dim, 
            'n_tweaks' : n_tweaks, 
            'interval' : interval, 
            'maxDomainValue' : 1, 
            'minDomainValue' : -1, 
            'temperature' : 1000, 
            'temperatureDecrease' : 100, 
            'maximize' : False ,
            'num_individuals': configuracions[1][0],
            'percentage_best': configuracions[1][1],
            'mutation_prob': configuracions[1][2]
            },{
            'func' : eq.F5,
            'maxStep' : 5,
            'n_experiments': n_experiments,
            'n_runs' : n_runs, 
            'n_dimentions' : n_dim, 
            'n_tweaks' : n_tweaks, 
            'interval' : interval, 
            'maxDomainValue' : 10, 
            'minDomainValue' : -10, 
            'temperature' : 1000, 
            'temperatureDecrease' : 100, 
            'maximize' : False ,
            'num_individuals': configuracions[3][0],
            'percentage_best': configuracions[3][1],
            'mutation_prob': configuracions[3][2]
            },{
            'func' : eq.F11,
            'maxStep' : 5,
            'n_experiments': n_experiments,
            'n_runs' : n_runs, 
            'n_dimentions' : n_dim, 
            'n_tweaks' : n_tweaks, 
            'interval' : interval, 
            'maxDomainValue' : 10, 
            'minDomainValue' : -10, 
            'temperature' : 1000, 
            'temperatureDecrease' : 0.01, 
            'maximize' : False ,
            'num_individuals': configuracions[4][0],
            'percentage_best': configuracions[4][1],
            'mutation_prob': configuracions[4][2]
            }]
        nombre = f'N({n_dim})'
        stats = Experiment.concat_stats(parameters_list, funciones,titulo_grafica)
        
        stats.to_csv(f'csv/{nombre}.csv',index=False)
        stats.to_latex(f'tex/{nombre}.tex', index=False)
        return stats

    @staticmethod
    def experiment(n_experiments, n_runs):
        try:
            os.remove('stats.csv')
            print("El archivo stats.csv ha sido eliminado correctamente.")
        except FileNotFoundError:
            print("El archivo stats.csv no existe.")
        except PermissionError:
            print("No tienes permisos para eliminar el archivo.")
        except Exception as e:
            print(f"Ocurrió un error al intentar borrar el archivo: {e}")

            
        #Experiment.experiment1dim(n_experiments, n_runs, '1 Dimension')
        
        
        configuraciones = [(150, 0.1, 0.1), (300, 0.1, 0.01), (50, 0.7, 0.1), (100, 0.2, 0.1)]
        
        Experiment.experiment2dim(n_experiments, n_runs, '2 Dimensiones', configuraciones)
        
        #Experiment.experimentNdim(2, n_experiments, n_runs, 'N(2) Dimensiones')
        #Experiment.experimentNdim(5,n_experiments, n_runs, 'N(5) Dimensiones')
        #Experiment.experimentNdim(10,n_experiments, n_runs, 'N(10) Dimensiones')
        #Experiment.experimentNdim(100,n_experiments, n_runs, 'N(100) Dimensiones')
        #Experiment.experimentNdim(1000,n_experiments, n_runs, 'N(1000) Dimensiones')  
        
        ruta_script = "highlighter.py"

        if os.path.exists(ruta_script):
            codigo_salida = os.system(f"python3 {ruta_script}")
            if codigo_salida == 0:
                print("Script ejecutado con éxito")
            else:
                print(f"Error al ejecutar (código {codigo_salida})")
        else:
            print("El archivo del script no existe en la ruta especificada")
        
              
  
    @staticmethod
    def convergencia(lista_convergencia, funciones, titulo_grafica):
        def plot_convergence(ax, data, title):
            for method, values in data.items():
                xd = values#[x / len(values) for x in values]  # Normalizar los valores
                ax.plot(xd, label=method)  # Añadir etiqueta para la leyenda, marker = 'o'
            ax.set_title(title)
            ax.set_xlabel("Iteraciones")
            ax.set_ylabel("Fitness")
        
        n_graficas = len(lista_convergencia)
                
        n_columnas = int(np.ceil(np.sqrt(n_graficas)))
        n_filas = int(np.ceil(n_graficas / n_columnas))
        
        fig, axs = plt.subplots(n_filas, n_columnas, figsize=(n_columnas * 5, n_filas * 4))
        
        # Si hay una sola gráfica, axs no es una lista, así que lo convertimos en una lista
        if n_graficas == 1:
            axs = [axs]
        else:
            axs = axs.flatten()

        for i in range(len(lista_convergencia)):
            ecuacion = lista_convergencia[i]
            for metodo in ecuacion:
                plot_convergence(axs[i], metodo, funciones[i].__name__)
        
        # Obtener handles y labels de la primera gráfica
        handles, labels_legend = axs[0].get_legend_handles_labels()
        
        # Crear una leyenda común fuera de las gráficas
        fig.legend(handles, labels_legend, loc='upper right', bbox_to_anchor=(1.1, 1.0))
        
        fig.suptitle(titulo_grafica, fontsize=16)

        
        # Ajustar el layout para evitar superposiciones
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Ajustar el espacio para la leyenda
        
        plt.savefig(f'fig/{titulo_grafica}.pdf')
        plt.show()