import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA

def get_feature_names_out(pipeline_step, input_features):
    """Helper function to get feature names from a pipeline step, typically PCA."""
    if hasattr(pipeline_step, 'get_feature_names_out'):
        return pipeline_step.get_feature_names_out(input_features)
    elif hasattr(pipeline_step, 'n_components_'):
        return [f"PC{i+1}" for i in range(pipeline_step.n_components_)]
    return [f"feature_{i}" for i in range(len(input_features))]


def interpret_pca_components(pca_pipeline_step, feature_names, pipeline_name):
    """Prints interpretation of PCA components."""
    if not hasattr(pca_pipeline_step, 'components_'):
        print(f"PCA step in {pipeline_name} not fitted or not a PCA estimator.")
        return

    loadings = pca_pipeline_step.components_
    explained_variance_ratio = pca_pipeline_step.explained_variance_ratio_
    
    print(f"\n--- Interpretation for {pipeline_name} ---")
    for i, component_loadings in enumerate(loadings):
        print(f"\nPrincipal Component {i+1} (Explains {explained_variance_ratio[i]*100:.2f}% of variance):")
        loadings_df = pd.DataFrame({
            'Variable': feature_names,
            'Loading': component_loadings
        })
        loadings_df['Absolute_Loading'] = np.abs(loadings_df['Loading'])
        loadings_df = loadings_df.sort_values(by='Absolute_Loading', ascending=False)
        
        print(loadings_df[['Variable', 'Loading']].head(10)) # Display top 10 contributing variables
        
        # Suggesting a meaning based on top positive and negative loadings
        top_positive = loadings_df[loadings_df['Loading'] > 0].head(3)['Variable'].tolist()
        top_negative = loadings_df[loadings_df['Loading'] < 0].sort_values(by='Loading').head(3)['Variable'].tolist()
        
        meaning = f"  Represents a contrast between features like [{', '.join(top_positive)}] and features like [{', '.join(top_negative)}]."
        if not top_negative:
            meaning = f"  Primarily driven by high values in features like [{', '.join(top_positive)}]."
        elif not top_positive:
            meaning = f"  Primarily driven by low values in features like (or high values in their inverse if applicable) [{', '.join(top_negative)}]."
        print(meaning)

def main():
    print("Starting PCA transformation script...")

    # 1. Load Data
    df_original = pd.read_csv("../data/dataset.csv")
    print("Successfully loaded data/dataset.csv")
    df = df_original.copy()

    # 2. One-hot encode 'Tipo di località'
    if 'Tipo di località' in df.columns:
        df = pd.get_dummies(df, columns=['Tipo di località'], prefix='Localita')
        print("Performed one-hot encoding on 'Tipo di località'.")
    else:
        print("Warning: 'Tipo di località' column not found for one-hot encoding.")

    # 3. Define Feature Sets
    var_economiche_cols = [
        "Reddito Pro-Capite per l'anno di riferimento", "Reddito da fabbricati - Frequenza", 
        "Reddito da lavoro dipendente e assimilati - Frequenza", "Reddito da pensione - Frequenza", 
        "Reddito da lavoro autonomo (comprensivo dei valori nulli) - Frequenza", 
        "Reddito di spettanza dell'imprenditore in contabilita' ordinaria  (comprensivo dei valori nulli) - Frequenza", 
        "Reddito di spettanza dell'imprenditore in contabilita' semplificata (comprensivo dei valori nulli) - Frequenza", 
        "Reddito complessivo minore o uguale a zero euro - Frequenza", 
        "Reddito complessivo da 0 a 10000 euro - Frequenza", 
        "Reddito complessivo da 10000 a 15000 euro - Frequenza", 
        "Reddito complessivo da 15000 a 26000 euro - Frequenza", 
        "Reddito complessivo da 26000 a 55000 euro - Frequenza", 
        "Reddito complessivo da 55000 a 75000 euro - Frequenza", 
        "Reddito complessivo da 75000 a 120000 euro - Frequenza", 
        "Reddito complessivo oltre 120000 euro - Frequenza"
    ]

    var_turistiche_cols = [
        'Esercizi a 5 stelle', 'Letti', 'Camere', 'Bagni', 'Esercizi a 4 stelle', 'Letti.1', 
        'Camere.1', 'Bagni.1', 'Esercizi a 3 stelle', 'Letti.2', 'Camere.2', 'Bagni.2', 
        'Esercizi a 2 stelle', 'Letti.3', 'Camere.3', 'Bagni.3', 'Esercizi a 1 stella', 
        'Letti.4', 'Camere.4', 'Bagni.4', 'Esercizi turistico-alberghiere', 'Letti.5', 
        'Camere.5', 'Bagni.5', 'Esercizi alberghieri', 'Letti.6', 'Camere.6', 'Bagni.6', 
        'Numero campeggi e villaggi turistici', 'Letti.7', 
        'Numero di alloggi in affitto gestiti in forma imprenditoriale', 'Letti.8', 
        'Numero di agriturismi', 'Letti.9', 'Numero di ostelli per la gioventù', 
        'Letti.10', 'Numero di case per ferie', 'Letti.11', 'Numero di rifugi alpini', 
        'Letti.12', 'Numero di altri esercizi ricettivi', 'Letti.13', 'Numero di b&b', 
        'Letti.14', 'Numero di esercizi extra-alberghieri', 'Letti.15', 'Numero', 'Letti.16'
    ]

    # Filter out columns that might not exist in the loaded df to prevent KeyErrors
    var_economiche_cols = [col for col in var_economiche_cols if col in df.columns]
    var_turistiche_cols = [col for col in var_turistiche_cols if col in df.columns]
    
    print(f"Found {len(var_economiche_cols)} economic variables for PCA.")
    print(f"Found {len(var_turistiche_cols)} touristic variables for PCA.")

    # 4. Create Pipelines
    # Economic Pipeline
    pipeline_econ = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('transformer', PowerTransformer(method='yeo-johnson', standardize=False)), # standardize=False as StandardScaler is next
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.90)) # Retain 90% of variance
    ])

    # Touristic Pipeline
    pipeline_tur = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('transformer', PowerTransformer(method='yeo-johnson', standardize=False)),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.80)) # Retain 80% of variance
    ])

    # 5. Apply Pipelines
    df_econ_pca_transformed = None
    df_tur_pca_transformed = None
    
    if var_economiche_cols:
        print("\nProcessing economic variables...")
        econ_data = df[var_economiche_cols]
        econ_pca_result = pipeline_econ.fit_transform(econ_data)
        
        # Get feature names for economic PCs
        econ_pca_step = pipeline_econ.named_steps['pca']
        econ_pc_names = [f"PC_Econ_{i+1}" for i in range(econ_pca_step.n_components_)]
        
        df_econ_pca_transformed = pd.DataFrame(econ_pca_result, columns=econ_pc_names, index=df.index)
        print(f"Economic PCA resulted in {econ_pca_step.n_components_} components.")
        interpret_pca_components(econ_pca_step, var_economiche_cols, "Economic Variables PCA")
    else:
        print("Skipping economic PCA as no economic variables were found.")

    if var_turistiche_cols:
        print("\nProcessing touristic variables...")
        tur_data = df[var_turistiche_cols]
        tur_pca_result = pipeline_tur.fit_transform(tur_data)
        
        # Get feature names for touristic PCs
        tur_pca_step = pipeline_tur.named_steps['pca']
        tur_pc_names = [f"PC_Tur_{i+1}" for i in range(tur_pca_step.n_components_)]

        df_tur_pca_transformed = pd.DataFrame(tur_pca_result, columns=tur_pc_names, index=df.index)
        print(f"Touristic PCA resulted in {tur_pca_step.n_components_} components.")
        interpret_pca_components(tur_pca_step, var_turistiche_cols, "Touristic Variables PCA")
    else:
        print("Skipping touristic PCA as no touristic variables were found.")

    # 6. Prepare Final DataFrame
    print("\nPreparing final DataFrame...")
    # Start with the original df (which now includes dummy variables if 'Tipo di località' was present)
    # Drop the original economic and touristic columns that were used in PCA
    cols_to_drop_from_original = list(set(var_economiche_cols + var_turistiche_cols))
    df_final = df.drop(columns=cols_to_drop_from_original, errors='ignore')

    # Concatenate with new PCA components
    if df_econ_pca_transformed is not None:
        df_final = pd.concat([df_final, df_econ_pca_transformed], axis=1)
    if df_tur_pca_transformed is not None:
        df_final = pd.concat([df_final, df_tur_pca_transformed], axis=1)
    
    print(f"Final DataFrame shape: {df_final.shape}")
    print("Final DataFrame columns:", df_final.columns.tolist())

    # 7. Save Final DataFrame
    output_path = "data/df_transformed_pca.csv"
    try:
        df_final.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved transformed data to {output_path}")
    except Exception as e:
        print(f"Error saving final DataFrame to {output_path}: {e}")

    print("\nPCA transformation script finished.")

if __name__ == "__main__":
    main()
