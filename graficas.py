import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import os

def recuperar_f1_scores(ruta_excel, pestanas=None):
    """
    Recupera los valores de F1-Score de un archivo Excel con múltiples pestañas.
    
    Args:
        ruta_excel (str): Ruta al archivo Excel.
        pestanas (list): Lista de pestañas a procesar. Si es None, procesa todas.
    
    Returns:
        dict: Diccionario anidado con la estructura {nombre_pestaña: {modelo_dimension: media_f1}}
    """
    try:
        xls = pd.ExcelFile(ruta_excel)
        resultado = {}
        
        # Si no especificas pestañas, usa todas disponibles
        sheets_a_procesar = pestanas if pestanas else xls.sheet_names
        
        for sheet_name in sheets_a_procesar:
            if sheet_name not in xls.sheet_names:
                print(f"[ADVERTENCIA] Pestaña '{sheet_name}' no encontrada en el Excel")
                continue
                
            df = pd.read_excel(xls, sheet_name=sheet_name)
            sheet_dict = {}
            
            # Verificar que la columna 'Modelos' existe
            if 'Modelos' not in df.columns:
                print(f"[ADVERTENCIA] Pestaña '{sheet_name}' no tiene columna 'Modelos'")
                continue
            
            for _, row in df.iterrows():
                modelo_dim = row['Modelos']
                if pd.isna(modelo_dim) or str(modelo_dim).strip() == '':
                    continue
                modelo_dim = str(modelo_dim).strip()
                
                if pd.isna(row['F1-Score']):
                    continue
                    
                f1_str = str(row['F1-Score'])
                
                # Parsear la cadena F1-Score (formato: "0: 0.72\n1: 0.92")
                valores = []
                for line in f1_str.split('\n'):
                    line = line.strip()
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            try:
                                val = float(parts[1].strip())
                                valores.append(val)
                            except ValueError:
                                continue
                
                # Calcular la media
                if valores:
                    media = sum(valores) / len(valores)
                    sheet_dict[modelo_dim] = media
            
            resultado[sheet_name] = sheet_dict
        
        print(f"[EXITO] Se recuperaron datos de {len(resultado)} pestañas")
        return resultado
    except FileNotFoundError:
        print(f"[ERROR] Archivo no encontrado: {ruta_excel}")
        return {}
    except Exception as e:
        print(f"[ERROR] Error al leer Excel: {e}")
        import traceback
        traceback.print_exc()
        return {}

def crear_grafica_comparativa(datos):
    """
    Crea una gráfica de barras comparativa de los F1-Scores, agrupada por dimensión.
    
    Args:
        datos (dict): Diccionario devuelto por recuperar_f1_scores.
    """
    df_list = []
    for algoritmo, inner_dict in datos.items():
        for modelo_dim, f1_score in inner_dict.items():
            try:
                modelo, dimension = modelo_dim.split(' ', 1)
            except ValueError:
                modelo = modelo_dim
                dimension = 'Unknown'
            df_list.append({
                'Algoritmo': algoritmo,
                'Modelo': modelo,
                'Dimension': dimension,
                'F1_Score': f1_score
            })
    
    df = pd.DataFrame(df_list)
    
    # Obtener los algoritmos únicos ordenados
    algoritmos = sorted(df['Algoritmo'].unique())
    
    # Obtener la paleta de colores
    colores = sns.color_palette('Set2', len(algoritmos))
    
    g = sns.catplot(
        data=df,
        kind='bar',
        x='Modelo',
        y='F1_Score',
        hue='Algoritmo',
        col='Dimension',
        col_wrap=2,
        palette='Set2',
        height=6,
        aspect=1.6,
        sharey=True,
        legend=False
    )
    g.set_titles('{col_name}')
    g.set_axis_labels('Modelo', 'F1-Score')
    
    # Establecer rango del eje Y y agregar grid horizontal
    for ax in g.axes.flat:
        ax.set_ylim(0, 1)  # Rango completo de 0 a 1
        ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=0.8)  # Grid horizontal punteado
        ax.set_axisbelow(True)  # Grid detrás de las barras
    
    g.figure.subplots_adjust(top=0.9, right=0.88, left=0.08, hspace=0.45, wspace=0.25)
    
    # Crear handles de leyenda con los colores correctos
    handles = [mpatches.Patch(color=colores[i], label=algo) for i, algo in enumerate(algoritmos)]
    g.figure.legend(
        handles=handles,
        title='Algoritmo',
        loc='center right',
        bbox_to_anchor=(1, 0.42),
        frameon=True,
        ncol=1
    )
    plt.suptitle('Comparación de F1-Score por Dimensión, Modelo y Algoritmo de Balanceo', y=0.96)
    g.figure.tight_layout(rect=[0, 0, 0.92, 0.94])
    g.figure.savefig('comparacion_f1_scores_por_dimension.png', dpi=500)
    plt.close(g.figure)
    print("[EXITO] Gráfica comparativa agrupada por dimensión guardada como 'comparacion_f1_scores_por_dimension.png'")

def generar_graficas_mbti(ruta_csv="dataset9K/MBTI_limpio.csv", columna_etiqueta="type"):
    print(f"[INFO] Leyendo dataset desde {ruta_csv}...")
    df = pd.read_csv(ruta_csv)
    
    colores = plt.cm.tab20(np.linspace(0, 1, 16))
    conteo_16 = df[columna_etiqueta].value_counts()
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(conteo_16.index, conteo_16.values, color=colores, edgecolor='black')
    
    plt.title('Distribución de las 16 Personalidades (MBTI)', fontsize=16)
    plt.xlabel('Tipo de Personalidad', fontsize=12)
    plt.ylabel('Posts', fontsize=12)
    plt.xticks()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Etiquetar los valores exactos encima de cada barra
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.01), int(yval), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('grafica_16_clases.png', dpi=500)
    plt.close()
    print("[EXITO] Gráfica de 16 clases guardada como 'grafica_16_clases.png'")

def generar_graficas_circulares(ruta_csv="dataset9K/MBTI_limpio.csv", columna_etiqueta="type"):
    print(f"[INFO] Leyendo dataset desde {ruta_csv}...")
    df = pd.read_csv(ruta_csv)
    conteo_16 = df[columna_etiqueta].value_counts()
    
    # Extraemos cuántas veces aparece cada letra en todo el dataset
    dim_counts = {'E': 0, 'I': 0, 'S': 0, 'N': 0, 'T': 0, 'F': 0, 'J': 0, 'P': 0}
    for tipo, cantidad in conteo_16.items():
        for letra in tipo:
            dim_counts[letra] += cantidad

    # Definimos la información para generar los 4 subgráficos
    # (Nombre 1, Nombre 2, Clave 1, Clave 2, Colores)
    pares = [
        ('Extraversión (E)', 'Introversión (I)', 'E', 'I', ['#ff9999', '#66b3ff']),
        ('Sensación (S)', 'Intuición (N)', 'S', 'N', ['#ffcc99', '#99ff99']),
        ('Pensamiento (T)', 'Sentimiento (F)', 'T', 'F', ['#c2c2f0', '#ffb3e6']),
        ('Juicio (J)', 'Percepción (P)', 'J', 'P', ['#ffb366', '#c2f0c2'])
    ]

    # Crear una figura general con 2 filas y 2 columnas (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Las 4 Dimensiones del MBTI', fontsize=18, fontweight='bold', y=0.98)

    for i, (label1, label2, clave1, clave2, colores) in enumerate(pares):
        fila = i // 2
        columna = i % 2
        ax = axs[fila, columna]
        
        valores = [dim_counts[clave1], dim_counts[clave2]]
        
        etiquetas = [f'{label1}\n({valores[0]})', f'{label2}\n({valores[1]})']
        
        _, _, autotexts = ax.pie(
            valores, 
            labels=etiquetas, 
            colors=colores, 
            autopct='%1.1f%%',       
            startangle=90,           
            explode=[0.05, 0],       
            shadow=True,             
            textprops={'fontsize': 13}
        )
        
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('black')
            
        ax.set_title(f'Eje {clave1} / {clave2}', fontsize=14, fontweight='bold', pad=15)

    # Ajustar el espaciado para que los gráficos no se superpongan
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('grafica_4_dimensiones_pie.png', dpi=500)
    plt.close()
    print("[EXITO] Gráfica de 4 dimensiones en círculos guardada como 'grafica_4_dimensiones_pie.png'")



if __name__ == "__main__":
    #generar_graficas_mbti()
    #generar_graficas_circulares()
    ruta_excel = os.path.join("resultados", "Comparativa_Maestra.xlsx")
    pestanas = ["SMOTE", "BorderlineSMOTE", "ADASYN", "SMOTETomek"]
    
    print(f"[INFO] Leyendo archivo: {ruta_excel}")
    print(f"[INFO] Pestañas a procesar: {pestanas}")
    dataset = recuperar_f1_scores(ruta_excel, pestanas=pestanas)
    if dataset:
        crear_grafica_comparativa(dataset)
    else:
        print("[ERROR] No se pudieron recuperar los datos del Excel.")
    pass