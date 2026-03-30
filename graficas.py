import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

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
    # ¡Asegúrate de poner la ruta correcta a tu CSV original aquí!
    #generar_graficas_mbti()
    generar_graficas_circulares()