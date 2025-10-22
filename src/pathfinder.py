# pathfinder.py
"""
Service for finding optimal flight paths using Graph Neural Networks.
"""

import logging
import pandas as pd
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data

from config import EDGE_DIST_THRESHOLD
from utils import haversine_distance
from gnn_models import SimpleGCN


class GNNPathFinder:
    def __init__(self, edge_threshold=EDGE_DIST_THRESHOLD):
        logging.info("Initializing GNNPathFinder.")
        self.edge_threshold = edge_threshold

    def build_graph_from_df(self, df: pd.DataFrame):
        logging.info("Building graph from DataFrame.")
        features = df[['longitude', 'latitude', 'weather_risk', 'bird_risk']].to_numpy(dtype=np.float32)
        coords = df[['longitude', 'latitude']].to_numpy(dtype=np.float32)
        edge_list = []
        edge_attr = []
        num_nodes = len(coords)
        
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                dist = haversine_distance((coords[i][0], coords[i][1]), (coords[j][0], coords[j][1]))
                if dist <= self.edge_threshold:
                    risk_i = (df.iloc[i]['weather_risk'] or 0) + (df.iloc[i]['bird_risk'] or 0)
                    risk_j = (df.iloc[j]['weather_risk'] or 0) + (df.iloc[j]['bird_risk'] or 0)
                    risk = (risk_i + risk_j) / 2.0
                    edge_list.append([i, j])
                    edge_list.append([j, i])
                    edge_attr.append(risk)
                    edge_attr.append(risk)
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty(0, dtype=torch.float)
        
        data = Data(x=torch.tensor(features, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
        return data

    def build_networkx_graph(self, data: Data, df: pd.DataFrame, embeddings: np.ndarray):
        G = nx.Graph()
        for i, row in df.iterrows():
            G.add_node(i, longitude=row['longitude'], latitude=row['latitude'],
                       weather_risk=row['weather_risk'], bird_risk=row['bird_risk'])
        
        edge_index_np = data.edge_index.numpy()
        for idx in range(edge_index_np.shape[1]):
            i = edge_index_np[0, idx]
            j = edge_index_np[1, idx]
            risk = data.edge_attr[idx].item() if data.edge_attr.numel() > 0 else 0
            weight = haversine_distance(
                (G.nodes[i]['longitude'], G.nodes[i]['latitude']),
                (G.nodes[j]['longitude'], G.nodes[j]['latitude'])
            ) + risk
            weight += 0.5 * np.linalg.norm(embeddings[i] - embeddings[j])
            G.add_edge(i, j, weight=weight)
        return G

    def find_optimal_path(self, df: pd.DataFrame, start_coord: tuple, end_coord: tuple):
        data = self.build_graph_from_df(df)
        if data.edge_index.numel() == 0:
            raise Exception("Graph has no edges; cannot find path.")
        
        model = SimpleGCN(in_channels=4, hidden_channels=8, out_channels=4)
        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index, data.edge_attr).numpy()
        
        G = self.build_networkx_graph(data, df, embeddings)

        def find_closest_node(target):
            return min(range(len(df)), key=lambda i: haversine_distance(
                (df.iloc[i]['longitude'], df.iloc[i]['latitude']), target))
        
        source_node = find_closest_node(start_coord)
        target_node = find_closest_node(end_coord)
        
        path = nx.dijkstra_path(G, source_node, target_node, weight='weight')
        total_cost = nx.dijkstra_path_length(G, source_node, target_node, weight='weight')
        return path, total_cost, df