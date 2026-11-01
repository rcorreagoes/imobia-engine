import os
import shutil

import kagglehub


def download_housing_dataset():
    print("🚀 A descarregar a versão mais recente do dataset do Kaggle...")
    
    # Faz o download usando o kagglehub
    download_path = kagglehub.dataset_download("luvathoms/portugal-real-estate-2024") # Ajuste se necessário para o nome exato
    print(f"📁 Ficheiros descarregados temporariamente em: {download_path}")

    # Definir a pasta de destino no seu projeto
    raw_data_dir = "data/raw"
    os.makedirs(raw_data_dir, exist_ok=True)

    # O kagglehub baixa para uma pasta cache. Vamos encontrar o arquivo CSV lá dentro
    csv_file_name = None
    for root, dirs, files in os.walk(download_path):
        for file in files:
            if file.endswith(".csv"):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(raw_data_dir, "portugal_housing.csv")
                
                # Copia o CSV para a sua pasta do projeto
                shutil.copy(source_file, destination_file)
                print(f"✨ Dataset copiado com sucesso para: {destination_file}")
                return destination_file

    print("⚠️ Nenhum arquivo CSV foi encontrado na pasta descarregada.")
    return None

if __name__ == "__main__":
    download_housing_dataset()