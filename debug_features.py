import os
import pandas as pd
from PIL import Image
import glob

def debug_feature_matching():
    """Debug para verificar correspondência entre imagens e features"""
    
    # Carregar CSV de features
    features_csv = "Dataset/radiomics_features.csv"
    print(f"📑 Carregando features de: {features_csv}")
    features_df = pd.read_csv(features_csv)
    print(f"✅ CSV carregado com {len(features_df)} linhas")
    print(f"📊 Colunas: {list(features_df.columns)}")
    
    # Verificar a última coluna (que deve ser o caminho da imagem)
    print(f"\n🔍 Última coluna: {features_df.columns[-1]}")
    print(f"🔍 Penúltima coluna: {features_df.columns[-2]}")
    
    # Verificar algumas linhas para entender a estrutura
    print(f"\n📋 Primeiras 3 linhas (últimas 2 colunas):")
    for i in range(min(3, len(features_df))):
        last_col = features_df.iloc[i, -1]
        second_last_col = features_df.iloc[i, -2]
        print(f"  Linha {i+1}: {second_last_col} | {last_col}")
    
    # Verificar se a penúltima coluna contém caminhos de imagem
    image_path_col = features_df.iloc[:, -2].tolist()
    print(f"\n🔍 Verificando penúltima coluna (image_path)...")
    print(f"  Primeiros 3 valores: {image_path_col[:3]}")
    
    # Verificar se há caminhos de imagem
    image_paths = []
    for value in image_path_col:
        if isinstance(value, str) and ('Dataset' in value or '.JPG' in value or '.jpg' in value):
            image_paths.append(value)
    
    print(f"✅ Encontrados {len(image_paths)} caminhos de imagem na última coluna")
    if image_paths:
        print(f"📝 Primeiros 3 caminhos:")
        for path in image_paths[:3]:
            print(f"  - {path}")
    
    # Encontrar imagens no dataset
    dataset_path = "Dataset/processed"
    print(f"\n📁 Procurando imagens em: {dataset_path}")
    
    # Encontrar todas as imagens
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    
    for ext in image_extensions:
        pattern = os.path.join(dataset_path, "**", ext)
        all_images.extend(glob.glob(pattern, recursive=True))
    
    print(f"✅ Encontradas {len(all_images)} imagens")
    
    # Verificar correspondência
    print(f"\n🔍 Verificando correspondência...")
    matched = 0
    not_matched = 0
    
    for i, img_path in enumerate(all_images[:5]):  # Primeiras 5 imagens
        print(f"\n--- Imagem {i+1} ---")
        print(f"📸 Caminho da imagem: {img_path}")
        
        # Tentar diferentes formatos de caminho
        found = False
        
        # 1. Caminho absoluto
        if img_path in image_path_col:
            print(f"✅ Encontrada (absoluto)")
            found = True
        else:
            print(f"❌ Não encontrada (absoluto)")
        
        # 2. Caminho relativo
        rel_path = os.path.relpath(img_path, os.path.dirname(dataset_path))
        rel_path = rel_path.replace('\\', '/')
        print(f"🔍 Caminho relativo: {rel_path}")
        
        if rel_path in image_path_col:
            print(f"✅ Encontrada (relativo)")
            found = True
        else:
            print(f"❌ Não encontrada (relativo)")
        
        # 3. Caminho sem prefixo Dataset/
        normalized_path = img_path.replace('Dataset\\', '').replace('Dataset/', '')
        print(f"🔍 Caminho normalizado: {normalized_path}")
        
        if normalized_path in image_path_col:
            print(f"✅ Encontrada (normalizado)")
            found = True
        else:
            print(f"❌ Não encontrada (normalizado)")
        
        if found:
            matched += 1
        else:
            not_matched += 1
            print(f"⚠️  Nenhum formato funcionou para: {img_path}")
    
    print(f"\n📊 Resumo:")
    print(f"✅ Correspondências encontradas: {matched}")
    print(f"❌ Sem correspondência: {not_matched}")

if __name__ == "__main__":
    debug_feature_matching() 