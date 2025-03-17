import os
import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

class ERDiagramGenerator:
    def __init__(self, csv_directory=None):
        """
        Inizializza il generatore di diagrammi ER
        
        Args:
            csv_directory (str): Percorso della directory contenente i file CSV
        """
        self.csv_directory = csv_directory or '.'
        self.tables = {}  # Dizionario con nome_tabella: [colonne]
        self.relationships = []  # Lista di tuple (tabella_origine, tabella_destinazione, tipo_relazione)
        self.primary_keys = {}  # Dizionario con nome_tabella: primary_key
    
    def load_csv_files(self, csv_directory=None):
        """
        Carica tutti i file CSV dalla directory specificata
        
        Args:
            csv_directory (str): Percorso della directory contenente i file CSV
        
        Returns:
            dict: Dizionario delle tabelle con le relative colonne
        """
        if csv_directory:
            self.csv_directory = csv_directory
        
        print(f"Ricerca file CSV in: {self.csv_directory}")
        
        # Trova tutti i file CSV nella directory
        csv_files = [f for f in os.listdir(self.csv_directory) if f.endswith('.csv')]
        print(f"Trovati {len(csv_files)} file CSV: {csv_files}")
        
        # Leggi ogni file CSV e memorizza la struttura
        for csv_file in csv_files:
            file_path = os.path.join(self.csv_directory, csv_file)
            table_name = os.path.splitext(csv_file)[0]
            
            try:
                # Leggi solo l'intestazione per ottenere i nomi delle colonne
                df = pd.read_csv(file_path, nrows=0)
                columns = list(df.columns)
                
                # Memorizza la tabella e le sue colonne
                self.tables[table_name] = columns
                
                # Identifica possibili chiavi primarie
                primary_key_candidates = [col for col in columns if 
                                         col.lower() == 'id' or 
                                         col.lower() == f'{table_name.lower()}_id' or
                                         col.lower() == 'id_' + table_name.lower()]
                
                if primary_key_candidates:
                    self.primary_keys[table_name] = primary_key_candidates[0]
                    
                print(f"Caricata tabella '{table_name}' con {len(columns)} colonne")
                
            except Exception as e:
                print(f"Errore durante la lettura del file {csv_file}: {e}")
        
        return self.tables
    
    def detect_relationships(self):
        """
        Rileva possibili relazioni tra le tabelle basandosi sui nomi delle colonne
        
        Returns:
            list: Lista di relazioni rilevate
        """
        self.relationships = []
        
        # Per ogni tabella
        for source_table, columns in self.tables.items():
            # Per ogni colonna nella tabella
            for column in columns:
                # Cerca possibili relazioni con altre tabelle
                for target_table in self.tables.keys():
                    # Evita l'auto-relazione se le colonne sono identiche
                    if source_table == target_table and column == self.primary_keys.get(target_table):
                        continue
                    
                    # Verifica se la colonna potrebbe essere una chiave esterna
                    if (column.lower() == f'{target_table.lower()}_id' or
                        column.lower() == target_table.lower() + '_id' or
                        (target_table.lower().endswith('s') and column.lower() == target_table.lower()[:-1] + '_id') or
                        column.lower() == 'id_' + target_table.lower()):
                        
                        # Aggiungi la relazione
                        self.relationships.append({
                            'source': source_table,
                            'target': target_table, 
                            'source_column': column,
                            'target_column': self.primary_keys.get(target_table, 'id'),
                            'type': 'many-to-one'  # Assumi che sia una relazione many-to-one
                        })
                        
                        print(f"Rilevata possibile relazione: {source_table}.{column} -> {target_table}.{self.primary_keys.get(target_table, 'id')}")
        
        return self.relationships
    
    def generate_diagram(self, output_file='er_diagram.png', figsize=(12, 10)):
        """
        Genera il diagramma ER usando NetworkX e Matplotlib
        
        Args:
            output_file (str): Nome del file di output per il diagramma
            figsize (tuple): Dimensioni della figura in pollici
            
        Returns:
            None
        """
        if not self.tables:
            print("Nessuna tabella caricata. Usa prima load_csv_files().")
            return
        
        if not self.relationships:
            print("Nessuna relazione rilevata. Usa prima detect_relationships().")
            self.detect_relationships()
        
        # Crea un grafo diretto
        G = nx.DiGraph()
        
        # Aggiungi i nodi (tabelle)
        for table_name, columns in self.tables.items():
            # Formatta le colonne come stringa
            columns_str = '\n'.join(columns)
            G.add_node(table_name, columns=columns_str)
        
        # Aggiungi gli archi (relazioni)
        for rel in self.relationships:
            G.add_edge(
                rel['source'], 
                rel['target'], 
                source_col=rel['source_column'],
                target_col=rel['target_column'],
                relationship=rel['type']
            )
        
        # Calcola un layout per il grafo
        pos = nx.spring_layout(G, k=2.0, iterations=50)
        
        # Crea la figura
        plt.figure(figsize=figsize)
        
        # Disegna i nodi (tabelle)
        for node in G.nodes():
            # Crea un rettangolo per rappresentare la tabella
            x, y = pos[node]
            width, height = 0.2, 0.15
            
            # Calcola le coordinate del rettangolo
            rect = FancyBboxPatch(
                (x - width/2, y - height/2),
                width, height,
                boxstyle="round,pad=0.3",
                facecolor='lightblue',
                edgecolor='blue',
                alpha=0.8
            )
            plt.gca().add_patch(rect)
            
            # Aggiungi il nome della tabella
            plt.text(x, y, node, fontsize=12, ha='center', va='center', fontweight='bold')
            
            # Aggiungi gli attributi (colonne) sotto al nome della tabella
            columns = G.nodes[node]['columns'].split('\n')
            column_text = '\n'.join(columns[:5])  # Limita a 5 colonne per leggibilità
            if len(columns) > 5:
                column_text += f"\n... ({len(columns)-5} altre)"
                
            plt.text(x, y - height/2 - 0.05, column_text, fontsize=8, ha='center', va='top')
        
        # Disegna gli archi (relazioni)
        for u, v, data in G.edges(data=True):
            # Calcola punti di inizio e fine
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            # Disegna la freccia
            plt.arrow(
                x1, y1, 
                (x2-x1)*0.8, (y2-y1)*0.8,  # Non arrivare completamente al nodo
                head_width=0.03,
                head_length=0.05,
                fc='black', 
                ec='black',
                length_includes_head=True
            )
            
            # Aggiungi etichetta alla relazione
            label = f"{data['source_col']} → {data['target_col']}"
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            plt.text(mid_x, mid_y, label, fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
        
        plt.axis('off')  # Nascondi gli assi
        plt.title('Diagramma ER dei file CSV')
        plt.tight_layout()
        
        # Salva il diagramma
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Diagramma ER salvato come '{output_file}'")
        
        # Mostra il diagramma
        plt.show()

    def generate_mermaid_code(self, output_file='er_diagram.md'):
        """
        Genera codice Mermaid per il diagramma ER
        
        Args:
            output_file (str): Nome del file di output per il codice Mermaid
            
        Returns:
            str: Codice Mermaid generato
        """
        if not self.tables:
            print("Nessuna tabella caricata. Usa prima load_csv_files().")
            return ""
        
        if not self.relationships:
            print("Nessuna relazione rilevata. Usa prima detect_relationships().")
            self.detect_relationships()
        
        # Crea il codice Mermaid
        mermaid_code = "```mermaid\nerDiagram\n"
        
        # Aggiungi le entità (tabelle)
        for table_name, columns in self.tables.items():
            mermaid_code += f"    {table_name} {{\n"
            
            # Aggiungi le colonne
            for column in columns:
                # Determina il tipo di dati basandosi sul nome della colonna
                data_type = "string"
                if "id" in column.lower():
                    data_type = "int"
                elif any(word in column.lower() for word in ["date", "time", "created", "updated"]):
                    data_type = "datetime"
                elif any(word in column.lower() for word in ["price", "amount", "total", "cost"]):
                    data_type = "float"
                
                mermaid_code += f"        {data_type} {column}\n"
            
            mermaid_code += "    }\n"
        
        # Aggiungi le relazioni
        for rel in self.relationships:
            source = rel['source']
            target = rel['target']
            rel_type = rel['type']
            
            # Converti il tipo di relazione in notazione Mermaid
            mermaid_rel = "|o--o{" if rel_type == "many-to-one" else "--"
            
            mermaid_code += f"    {source} {mermaid_rel} {target} : \"has\"\n"
        
        mermaid_code += "```"
        
        # Salva il codice Mermaid in un file
        if output_file:
            with open(output_file, 'w') as f:
                f.write(mermaid_code)
            print(f"Codice Mermaid salvato come '{output_file}'")
        
        return mermaid_code


# Esempio di utilizzo
if __name__ == "__main__":
    # Crea l'istanza del generatore di diagrammi
    generator = ERDiagramGenerator()
    
    # Chiedi all'utente la directory dei file CSV
    csv_dir = input("Inserisci il percorso della directory contenente i file CSV (premi Invio per usare la directory corrente): ")
    if not csv_dir:
        csv_dir = "."
    
    # Carica i file CSV
    generator.load_csv_files(csv_dir)
    
    # Rileva le relazioni
    generator.detect_relationships()
    
    # Genera il diagramma ER
    generator.generate_diagram()
    
    # Genera anche il codice Mermaid
    mermaid_code = generator.generate_mermaid_code()
    print("\nCodice Mermaid generato:")
    print(mermaid_code)
