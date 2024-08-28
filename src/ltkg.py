import pandas as pd
import numpy as np
import torch
import conceptnet as cn 
# import lsin 
import networkx as nx
import matplotlib.pyplot as plt




icews14 = pd.read_csv("icews14.csv")

batch_size = 1000
entities = icews14["entity"].unique()
relations = icews14["relation"].unique()
timestamps = icews14["timestamp"].unique()


entity_knowledge = cn.query(entities)
relation_knowledge = cn.query(relations)


lsin_model = lsin.LSIN(entity_knowledge, relation_knowledge, hidden_size=128, num_layers=2, dropout=0.1)


optimizer = torch.optim.Adam(lsin_model.parameters(), lr=0.001)
loss_function = torch.nn.BCELoss()
epochs = 10

for epoch in range(epochs):
    
    icews14 = icews14.sample(frac=1)
    
    for i in range(0, len(icews14), batch_size):
        
        batch = icews14[i:i+batch_size]
        
        e_s = torch.tensor(batch["e_s"].values) 
        r = torch.tensor(batch["r"].values) 
        e_o = torch.tensor(batch["e_o"].values) 
        t = torch.tensor(batch["t"].values) 
        y = torch.tensor(batch["y"].values) 
        
        y_pred = lsin_model(e_s, r, e_o, t)
        
        loss = loss_function(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
new_events = []
for event in icews14:    
    e_s, r, e_o, t = event["e_s"], event["r"], event["e_o"], event["t"]    
    t_new = np.random.choice(timestamps[timestamps > t])    
    p_causal = lsin_model(e_s, r, e_o, t_new)
    if p_causal > 0.5:
        new_events.append((e_s, r, e_o, t_new))


graph = nx.DiGraph()

for entity in entities:
    graph.add_node(entity, **entity_knowledge[entity])
for relation in relations:
    graph.add_node(relation, **relation_knowledge[relation])

for event in icews14:
    e_s, r, e_o, t = event["e_s"], event["r"], event["e_o"], event["t"]
    graph.add_edge(e_s, r, timestamp=t)
    graph.add_edge(r, e_o, timestamp=t)
for event in new_events:
    e_s, r, e_o, t_new = event[0], event[1], event[2], event[3]
    graph.add_edge(e_s, r, time_range=(t,t_new))
    graph.add_edge(r, e_o, time_range=(t,t_new))
    
nx.write_gml(graph, "graph.gml")

# nx.draw(graph, node_color="red", edge_color="blue")
# plt.show()