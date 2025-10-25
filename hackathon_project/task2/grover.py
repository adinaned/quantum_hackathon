
# from qiskit import QuantumCircuit, transpile
# import itertools
# import networkx as nx
# import random
#
# # --- Folosim graful tău G_total, free_edges, occupied_edges ---
# n_edges = len(free_edges)
# G_total = nx.Graph()
# G_total.add_edges_from(free_edges + occupied_edges)
#
#
# # --- Helper: verifică drum valid ---
# def is_valid_path(selected_edges, G_total):
#     subG = nx.Graph()
#     subG.add_nodes_from(G_total.nodes())
#     subG.add_edges_from(selected_edges)
#
#     for node in subG.nodes():
#         if subG.degree(node) > 2:
#             return False
#     nodes_in_path = [v for e in selected_edges for v in e]
#     subG_nodes = subG.subgraph(nodes_in_path)
#     if nx.number_connected_components(subG_nodes) > 1:
#         return False
#     return True
#
#
# # --- Generăm combinații valide (pentru demo, clasic) ---
# max_roads = 6
# valid_combinations = []
#
# for length in range(max_roads, 0, -1):
#     candidates = itertools.combinations(range(n_edges), length)
#     for comb in candidates:
#         selected_edges = [free_edges[i] for i in comb]
#         if is_valid_path(selected_edges, G_total):
#             valid_combinations.append(comb)
#     if valid_combinations:
#         break  # găsim cea mai lungă lungime disponibilă
#
# print(f"Număr combinații valide: {len(valid_combinations)}, lungime: {length}")
#
# if n_edges == 0:
#     print("Nu există muchii libere disponibile. Nu se poate construi circuitul.")
# else:
#     qc = QuantumCircuit(n_edges, n_edges)
#     qc.h(range(n_edges))  # superpoziție uniformă
#     qc.measure(range(n_edges), range(n_edges))
#
# # Oracol simplu: marchează combinații valide (demo)
# # În practică, pentru 15 muchii poți folosi poarta X + MCT pe fiecare combinație validă
# # Aici simulăm Grover alegând aleator un candidat valid
# if valid_combinations:
#     best_path = random.choice(valid_combinations)
#     print("Sugestie quantum (simulat):")
#     for i in best_path:
#         print(free_edges[i])
