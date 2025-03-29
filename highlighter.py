import os
import re
import pandas as pd

def highlight_best_results(tex_file):
    with open(tex_file, 'r', encoding='utf-8') as file:
        content = file.readlines()
    
    table_data = []
    problem = None
    current_problem = None
    headers = None
    
    for line in content:
        line = line.strip()
        if line.startswith('\toprule') or line.startswith('\midrule') or line.startswith('\bottomrule'):
            continue
        
        if headers is None and 'Problema' in line:
            headers = [x.strip() for x in line.split('&')]
            headers = [h.replace('\\', '').strip() for h in headers]
            continue
        
        if '&' in line:
            parts = [x.strip().replace('\\', '') for x in line.split('&')]
            if re.match(r'^F\d+', parts[0]):
                problem = parts[0]
            else:
                parts.insert(0, problem)
            table_data.append(parts)
    
    df = pd.DataFrame(table_data, columns=headers)
    
    numeric_cols = ['Máximo', 'Mínimo', 'Mediana', 'IQR', 'Media', 'STD', 'Mejor Solución']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for problem in df['Problema'].unique():
        subset = df[df['Problema'] == problem]
        for col in numeric_cols:
            if col in df.columns:
                best_value = subset[col].min()
                df.loc[(df['Problema'] == problem) & (df[col] == best_value), col] = f'\\textbf{{{best_value}}}'
    
    new_content = []
    new_content.append("\\begin{tabular}{llrrrrrrr}\n")
    new_content.append("\\toprule\n")
    new_content.append(" & ".join(headers) + " \\\\ \n")
    new_content.append("\\midrule\n")
    
    current_problem = None
    for row in df.itertuples(index=False):
        if current_problem is None:
            current_problem = row[0]
        elif row[0] != current_problem:
            new_content.append("\\midrule\n")
            current_problem = row[0]
        
        new_line = ' & '.join(map(str, row)) + ' \\\\ \n'
        new_content.append(new_line)
    
    new_content.append("\\bottomrule\n")
    new_content.append("\\end{tabular}\n")
    
    with open('highlighted_' + tex_file, 'w', encoding='utf-8') as file:
        file.writelines(new_content)
    
    print(f'Archivo procesado y guardado como highlighted_{tex_file}')

# Uso: highlight_best_results('tu_archivo.tex')

tablas = ['UnaDimension', 'DosDimensiones', 'N(2)', 'N(5)', 'N(10)','N(100)','N(1000)']
for tabla in tablas:
    highlight_best_results(os.path.join('tex', f'{tabla}.tex'))