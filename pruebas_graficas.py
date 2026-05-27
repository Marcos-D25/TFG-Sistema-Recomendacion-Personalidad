import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Configuración
DATASET_PATH = "datasetRBFT"
MODELOS_PATH = "Modelos Definitivos"

# Modelos específicos de main.py
LISTA_MODELOS = [
    "Modelos Definitivos\\SMOTE\\E-I_Perceptron_Multicapa.joblib",
    "Modelos Definitivos\\BorderlineSMOTE\\S-N_XGBoost.joblib",
    "Modelos Definitivos\\SMOTETomek\\T-F_Linear_SVC.joblib",
    "Modelos Definitivos\\SMOTE\\J-P_Regresion_Logistica.joblib"
]

EJES = ["E-I", "S-N", "T-F", "J-P"]
EJES_INTERNAS = ["E/I", "S/N", "T/F", "J/P"]

def cargar_dataset_test(eje_interna: str) -> pd.DataFrame:
    """
    Carga el dataset para un eje específico y lo divide en train/test/val.
    Retorna test_df
    """
    # Mapear eje interno a archivo
    mapeo_archivos = {
        "E/I": "datasetEI.parquet",
        "S/N": "datasetSN.parquet",
        "T/F": "datasetTF.parquet",
        "J/P": "datasetJP.parquet"
    }
    
    archivo = mapeo_archivos[eje_interna]
    ruta = os.path.join(DATASET_PATH, archivo)
    
    print(f"[INFO] Cargando dataset desde {ruta}...")
    df = pd.read_parquet(ruta)
    df = df[[eje_interna, "Embedding"]].rename(columns={eje_interna: "MBTI"})
    
    # Dividir en train/test/val (80/10/10)
    X = df.drop(columns=["MBTI"])
    y = df["MBTI"]
    
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_eval, y_eval, test_size=0.50, random_state=42, stratify=y_eval
    )
    
    test_df = pd.concat([X_test, y_test], axis=1)
    print(f"✓ Dataset cargado: {len(test_df)} muestras en test")
    
    return test_df

def cargar_modelo(ruta_modelo: str):
    """
    Carga un modelo desde la ruta especificada.
    """
    ruta_corregida = ruta_modelo.replace("\\", os.sep)
    print(f"[INFO] Cargando modelo desde {ruta_corregida}...")
    
    if not os.path.exists(ruta_corregida):
        raise FileNotFoundError(f"Modelo no encontrado: {ruta_corregida}")
    
    modelo = joblib.load(ruta_corregida)
    print(f"✓ Modelo cargado exitosamente")
    return modelo

def generar_embeddings_roberta(embeddings_list):
    """
    Convierte la lista de embeddings a un array de numpy en formato float32.
    Los embeddings ya están en el dataframe, solo necesitan convertirse.
    """
    embeddings = []
    for emb in embeddings_list:
        if isinstance(emb, np.ndarray):
            embeddings.append(emb.astype(np.float32))
        elif isinstance(emb, list):
            embeddings.append(np.array(emb, dtype=np.float32))
        else:
            embeddings.append(np.array(emb, dtype=np.float32))
    
    return np.array(embeddings, dtype=np.float32)

def evaluar_modelo_eje(test_df, modelo, encoder_path, eje_idx: int) -> float:
    """
    Evalúa un modelo en un eje específico.
    Retorna el F1-Score macro.
    """
    print(f"\n[INFO] Evaluando eje {eje_idx} ({EJES[eje_idx]})...")
    
    try:
        # Obtener embeddings del dataframe
        print(f"  Obteniendo embeddings...")
        X_embeddings = generar_embeddings_roberta(test_df["Embedding"].tolist())
        
        y_test = test_df["MBTI"].tolist()
        
        # Predecir
        print(f"  Realizando predicciones...")
        y_pred = modelo.predict(X_embeddings)
        
        # Calcular F1-macro
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        print(f"  ✓ F1-Score (Macro): {f1_macro:.4f}")
        
        return f1_macro
        
    except Exception as e:
        print(f"  ✗ Error evaluando eje {eje_idx}: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def crear_grafica_f1_macro(resultados):
    """
    Crea una gráfica de barras con los F1-macro de cada eje.
    """
    print("\n[INFO] Generando gráfica de F1-Scores...")
    
    ejes = list(resultados.keys())
    f1_scores = list(resultados.values())
    
    # Crear gráfica
    plt.figure(figsize=(10, 6))
    colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    barras = plt.bar(ejes, f1_scores, color=colores, edgecolor='black', linewidth=2, alpha=0.8)
    
    # Añadir valores en las barras
    for barra in barras:
        altura = barra.get_height()
        plt.text(
            barra.get_x() + barra.get_width()/2, 
            altura + 0.01,
            f'{altura:.4f}',
            ha='center', 
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    plt.title('F1-Score Macro por Eje MBTI', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Eje MBTI', fontsize=13, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=13, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5, linewidth=1)
    plt.tight_layout()
    
    plt.savefig('grafica_f1_macro_por_eje.png', dpi=500, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada como 'grafica_f1_macro_por_eje.png'")

def generar_reporte(resultados):
    """
    Genera un reporte textual con los resultados.
    """
    print("\n" + "="*60)
    print("REPORTE DE EVALUACIÓN - F1-SCORE MACRO")
    print("="*60)
    
    for eje, f1 in resultados.items():
        print(f"{eje}: {f1:.4f}")
    
    media = np.mean(list(resultados.values()))
    print(f"\nMedia: {media:.4f}")
    print("="*60)

def main():
    """
    Función principal que ejecuta todo el pipeline.
    """
    print("\n" + "="*60)
    print("EVALUACIÓN DE MODELOS CON F1-SCORE MACRO")
    print("="*60)
    
    resultados = {}
    
    # Para cada eje, cargar datos, modelo y evaluar
    for eje_idx, (eje_sigla, eje_interna, modelo_path) in enumerate(
        zip(EJES, EJES_INTERNAS, LISTA_MODELOS)
    ):
        print(f"\n{'='*60}")
        print(f"Procesando eje {eje_idx}: {eje_sigla}")
        print(f"{'='*60}")
        
        try:
            # Cargar dataset de test
            test_df = cargar_dataset_test(eje_interna)
            
            # Cargar modelo
            modelo = cargar_modelo(modelo_path)
            
            # Evaluar
            f1_macro = evaluar_modelo_eje(test_df, modelo, None, eje_idx)
            resultados[eje_sigla] = f1_macro
            
        except Exception as e:
            print(f"[ERROR] Error procesando eje {eje_sigla}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generar salidas
    if resultados:
        crear_grafica_f1_macro(resultados)
        generar_reporte(resultados)
        
        print("\n" + "="*60)
        print("✅ PROCESO COMPLETADO EXITOSAMENTE")
        print("="*60)
    else:
        print("[ERROR] No se pudieron obtener resultados")

if __name__ == "__main__":
    main()
