from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import joblib
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

app = FastAPI(
    title="HabitaMetric ML Engine",
    description="API de Preço de Imóveis em Portugal baseada em Machine Learning",
    version="1.0.0"
)

# Caminhos dos artefatos
TRENDS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "district_yearly_trends.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "housing_model.pkl")
COLUMNS_PATH = os.path.join(PROJECT_ROOT, "models", "model_columns.pkl")
print(f"📁 A procurar o ficheiro de tendências em: {TRENDS_PATH}")

model = None
model_columns = None
trends_df = None

@app.on_event("startup")
def load_artifacts():
    global model, model_columns, trends_df
    
    # 1. Carregar Modelo de Predição Atual
    if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
        model = joblib.load(MODEL_PATH)
        model_columns = joblib.load(COLUMNS_PATH)
        print("✅ Modelo e colunas carregados com sucesso na memória!")
    else:
        print("⚠️ Atenção: Artefatos do modelo não encontrados. Execute o train.py primeiro.")

    # 2. Carregar Histórico de Tendências por Distrito
    if os.path.exists(TRENDS_PATH):
        trends_df = pd.read_csv(TRENDS_PATH)
        print("✅ Dados de tendências históricas carregados com sucesso!")
    else:
        print("⚠️ Atenção: Ficheiro district_yearly_trends.csv não encontrado. Execute o preprocessing.py.")
        
# Definir o contrato de dados que a API espera receber (ajuste conforme os campos principais)
class PropertyInput(BaseModel):
    District: str
    City: str
    Type: str
    TotalArea: float
    LivingArea: float
    NumberOfBedrooms: int
    NumberOfBathrooms: int
    Garage: str
    Elevator: str
    ConstructionYear: int
    EnergyCertificate: str

@app.get("/")
def health_check():
    return {"status": "online", "message": "Engine de Machine Learning em execução!"}

@app.post("/predict")
def predict_price(data: PropertyInput):
    if model is None or model_columns is None:
        raise HTTPException(status_code=500, detail="O modelo de Machine Learning não foi carregado na inicialização.")
    
    try:
        # 1. Converter o payload recebido em DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # 2. Aplicar o mesmo One-Hot Encoding usado no treino
        input_encoded = pd.get_dummies(input_data)
        
        # 3. Reindexar para garantir que todas as colunas do treino estejam presentes (faltantes viram 0)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
        
        # 4. Fazer a predição
        prediction = model.predict(input_encoded)
        estimated_price = float(prediction[0])
        
        return {
            "estimated_price_eur": round(estimated_price, 2),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar a predição: {str(e)}")
    
# =========================================================================
# NOVO ENDPOINT: Histórico de Valorização por Distrito
# =========================================================================
@app.get("/trends/{district}")
def get_district_trends(
    district: str, 
    municipality: str = Query(None, description="Filtrar por município/concelho"),
    property_type: str = Query(None, description="Filtrar por tipo de imóvel (ex: Apartment, House)")
):
    try:
        if trends_df is None:
            raise HTTPException(status_code=500, detail="Base de tendências não está carregada na memória.")
        
        # 1. Filtrar por Distrito
        filtered = trends_df[trends_df['District'].astype(str).str.lower().str.strip() == district.lower().strip()]
        if filtered.empty:
            raise HTTPException(status_code=404, detail=f"Distrito '{district}' não encontrado.")
        
        # 2. Filtrar por Município (se fornecido)
        if municipality:
            if 'Municipality' not in filtered.columns:
                raise HTTPException(status_code=500, detail="A coluna 'Municipality' não existe no CSV.")
            filtered = filtered[filtered['Municipality'].astype(str).str.lower().str.strip() == municipality.lower().strip()]
            if filtered.empty:
                raise HTTPException(status_code=404, detail=f"Município '{municipality}' não encontrado em '{district}'.")
        
        # 3. CENÁRIO A: Se o Tipo de Imóvel FOI especificado
        if property_type:
            if 'Type' not in filtered.columns:
                raise HTTPException(status_code=500, detail="A coluna 'Type' não existe no CSV.")
            filtered = filtered[filtered['Type'].astype(str).str.lower().str.strip() == property_type.lower().strip()]
            if filtered.empty:
                raise HTTPException(status_code=404, detail=f"Tipo de imóvel '{property_type}' não encontrado para os filtros selecionados.")
            
            grouped = filtered.groupby('PublishYear')['MedianPricePerSqm'].median().reset_index()
            result = grouped.rename(
                columns={'PublishYear': 'year', 'MedianPricePerSqm': 'median_price_sqm'}
            ).to_dict(orient='records')
            
            return {
                "district": district,
                "municipality": municipality or "All",
                "property_type": property_type,
                "historical_trend": result
            }
        
        # 4. CENÁRIO B: Se o Tipo de Imóvel NÃO foi especificado (Dividir por Tipo)
        else:
            if 'Type' not in filtered.columns:
                raise HTTPException(status_code=500, detail="A coluna 'Type' não existe no CSV.")
            
            # Agrupar por Ano E por Tipo de Imóvel
            grouped = filtered.groupby(['PublishYear', 'Type'])['MedianPricePerSqm'].median().reset_index()
            
            result = grouped.rename(
                columns={
                    'PublishYear': 'year', 
                    'MedianPricePerSqm': 'median_price_sqm',
                    'Type': 'property_type'
                }
            ).to_dict(orient='records')
            
            return {
                "district": district,
                "municipality": municipality or "All",
                "property_type": "All (Breakdown by Type)",
                "historical_trend_by_type": result
            }
                
        # ... (bloco except mantido igual)
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ ERRO CRÍTICO EM /trends/{district}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {str(e)}")
   
# Endpoint para obter a lista de distritos e municípios disponíveis
@app.get("/locations")
def get_locations():
    try:
        # Verifica se temos a base de dados disponível
        source_df = trends_df if trends_df is not None else (df_clean if 'df_clean' in globals() else None)
        
        if source_df is None:
            raise HTTPException(status_code=500, detail="Base de dados não está carregada na memória.")
        
        # Identifica a coluna correta de município
        municipality_col = 'Municipality' if 'Municipality' in source_df.columns else ('City' if 'City' in source_df.columns else 'Town')
        
        if 'District' not in source_df.columns or municipality_col not in source_df.columns:
            raise HTTPException(status_code=500, detail="Colunas de Distrito ou Município em falta no dataset.")
        
        # Agrupa os municípios por distrito, removendo duplicados e ordenando alfabeticamente
        locations_dict = {}
        grouped = source_df[['District', municipality_col]].dropna().drop_duplicates()
        
        for district, group in grouped.groupby('District'):
            municipalities = sorted(group[municipality_col].astype(str).unique().tolist())
            locations_dict[str(district)] = municipalities
            
        # Transforma num array de objetos estruturado (como você pensou)
        result = [
            {"district": district, "municipalities": concelhos}
            for district, concelhos in sorted(locations_dict.items())
        ]
        
        return {
            "total_districts": len(result),
            "locations": result
        }
        
    except Exception as e:
        print(f"❌ ERRO EM /locations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
    
@app.get("/property-types")
def get_property_types():
    try:
        source_df = trends_df if trends_df is not None else (df_clean if 'df_clean' in globals() else None)
        
        if source_df is None:
            raise HTTPException(status_code=500, detail="Base de dados não está carregada na memória.")
        
        # Identifica a coluna correta de tipo de imóvel (ex: 'Type' ou 'PropertyType')
        type_col = 'Type' if 'Type' in source_df.columns else ('PropertyType' if 'PropertyType' in source_df.columns else None)
        
        if not type_col:
            raise HTTPException(status_code=500, detail="Coluna de tipo de imóvel não encontrada no dataset.")
        
        # Extrai valores únicos, remove nulos e ordena alfabeticamente
        property_types = sorted(source_df[type_col].dropna().astype(str).unique().tolist())
        
        return {
            "total_types": len(property_types),
            "property_types": property_types
        }
        
    except Exception as e:
        print(f"❌ ERRO EM /property-types: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")