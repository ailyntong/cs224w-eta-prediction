from bisect import bisect, bisect_left
import json
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple

def load_graph(nodes_file: str, edges_file: str) -> nx.Graph:
  """
  Load input files into graph. 
  - Each line in `nodes_file` should have node_id, x_coord, y_coord
  - Each line in `edges_file` should have edge_id, node_id1, node_id2, length
  """

  G = nx.Graph()

  with open(nodes_file) as f:
    for line in f.readlines():
      # node id, x coordinate, y coordinate
      tokens = line.split(' ')
      v = int(tokens[0])
      x = float(tokens[1])
      y = float(tokens[2])
      G.add_node(v, pos=(x, y))

  with open(edges_file) as f:
    for line in f.readlines():
      # edge id, node id1, node id2, l2 distance
      tokens = line.split(' ')
      id = int(tokens[0])
      v1 = int(tokens[1])
      v2 = int(tokens[2])
      dist = float(tokens[3])
      G.add_edge(v1, v2, weight=dist, id=id)

  return G

def l2_distance(u: int, v: int, graph: nx.Graph = None, positions=None) -> float:
  """
  Calculates the L2 distance between nodes `u` and `v` in graph `G`.
  """
  assert graph or positions, "At least one of `graph` or `positions` must be valid"
  if positions is None:
    positions = nx.get_node_attributes(graph, 'pos')

  x1, y1 = positions[u]
  x2, y2 = positions[v]
  return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def generate_trips_fast(G: nx.Graph, num_srcs: int, trips_per_src: int, 
                        out_file: str) -> None:
  """
  Generate `trips_per_src` trips for `num_srcs` randomly chosen nodes.
  Trip lengths do NOT follow any distribution

  Output: File with tuples of (src id, dst id, L2 distance)
  """

  positions = nx.get_node_attributes(G, 'pos')

  with open(out_file, 'w+') as f:
    nodes = list(G.nodes)
    dists = []

    i = 0
    for u in np.random.choice(nodes, num_srcs):
      for v in np.random.choice(nodes, trips_per_src):
        dist = l2_distance(u, v, positions=positions)
        dists.append(dist)
        f.write(f"{u} {v} {dist}\n")
      i += trips_per_src
      if i % 100 == 0:
        print(f"{i} / {num_srcs * trips_per_src} complete")

  print(f"Average trip length: {sum(dists) / len(dists)}")

def generate_trips_boxselect(G: nx.Graph, num_trips: int, out_file: str) -> None:
  """
  Generate `num_trips` trips using an exponential distribution.

  Output: File with tuples of (src id, dst id, L2 distance)
  """

  min_trip_dist = 200
  beta = 1000
  
  positions = nx.get_node_attributes(G, 'pos')
  nodes = list(G.nodes)
  nodes_with_pos = sorted([(*positions[u], u) for u in G.nodes])

  # print(nodes_with_pos[0], nodes_with_pos[-1])

  with open(out_file, 'w+') as f:
    # For each randomly selected source node `u`, 
    # 1. Generate a target trip distance r = MIN_TRIP_DIST + Exp(1/beta)
    # 2. Identify all candidate destination nodes following the illustration below:
    # u ------ u_x+0.5r ------------- u_x+1.5r
    # |           |                  |
    # |           |                  |
    # u_y+0.5r ---+                  |
    # |                              |
    # |        CANDIDATE             |
    # |         REGION               |
    # |                              |
    # u_y+1.5r ---+------------------+
    #
    # 3. Attempt to select destination node `v` from whose L2 distance
    #    from `u` is within 5% of `r`
    #    - If no success after 10 attempts, take the bottom-right node 
    #      from candidate region

    for u in np.random.choice(nodes, num_trips):
      r = np.random.exponential(beta) + min_trip_dist
      # print(u, r)
      x, y = positions[u]
      i = bisect_left(nodes_with_pos, (x + 0.5 * r, y + 0.5 * r, u))
      j = bisect_left(nodes_with_pos, (x + 1.5 * r, y + 1.5 * r, -1))
      candidates = [t[2] for t in nodes_with_pos[i:j]]

      if len(candidates) == 0:
        continue

      for attempt in range(10): 
        v = np.random.choice(candidates)
        dist = l2_distance(u, v, positions=positions)
        if abs(r - dist) < 0.05 * r:
          f.write(f"{u} {v} {dist}\n")
          break

        if attempt == 9:
          v = candidates[-1]
          dist = l2_distance(u, v, positions=positions)
          f.write(f"{u} {v} {dist}\n")

def generate_trips(G: nx.Graph, num_trips: int, strategy='boxselect') -> str:
  """
  Generates `num_trips` trips using the given strategy.

  Return: Name of the file storing the trips.
  """

  trips_file = 'trips_' + strategy + '.txt'
  if strategy == 'fast':
    generate_trips_fast(G, num_trips/10, 10, trips_file)
  elif strategy == 'boxselect':
    generate_trips_boxselect(G, num_trips, trips_file)
  else:
    raise Exception('Invalid trip generation strategy')
  return trips_file

def format_segment(id: int, data: List) -> Dict:
  """
  Converts segment statistics into json (dictionary) format
  """
  total, count = data[0], data[1]
  avg_eta = 0 if count == 0 else total / count
  return {'id': id, 'average_eta': avg_eta}

def format_supersegment(id: int, eta: float, length: float, segment_ids: List) -> Dict:
  """
  Converts supersegment statistics into json (dictionary) format
  """
  return {'id': id, 'eta': eta, 'length': length, 'segment_ids': segment_ids}

def generate_traffic_simple(G: nx.Graph, s: int, t: int, segments, 
                            beta: float = 5, 
                            min_delta: float = -0.25, max_delta: float = 0.25,
                            mu: float = 50, sigma: float = 4) -> List:
  # print(f"Generating traffic for src={s}, dst={t}")
  
  num_trials = int(np.random.exponential(beta))
  default_path = nx.dijkstra_path(G, s, t)

  deltas = [np.random.random() * (max_delta - min_delta) + min_delta for _ in range(num_trials)]
  
  # Update segments
  for i in range(len(default_path) - 1):
    u, v = default_path[i], default_path[i+1]
    eid, eweight = G.edges[u, v]['id'], G.edges[u, v]['weight']

    segments[eid][0] += (num_trials + sum(deltas)) * eweight
    segments[eid][1] += num_trials

  # Create supersegments
  supersegments = []
  for delta in deltas:
    i = 0
    while i < len(default_path) - 1:
      num_segments = int(np.random.normal(loc=mu, scale=sigma))
      eta = 0
      length = 0
      segment_ids = []
      for _ in range(num_segments):
        if i == len(default_path) - 1:
          break
        u, v = default_path[i], default_path[i+1]
        eid, eweight = G.edges[u, v]['id'], G.edges[u, v]['weight']

        segment_ids.append(eid)
        eta += (1 + delta) * eweight
        length += eweight

        supersegments.append((eta, length, segment_ids))
        i += 1

  return supersegments


def generate_traffic_for_trips(G: nx.Graph,
                               trips_file: str, 
                               segments_file: str, supersegments_file: str,
                               strategy: str = 'simple') -> None:
  if strategy != 'simple':
    raise Exception("Invalid traffic generation strategy! Options: {'simple'}")
  else:
    traffic_fn = generate_traffic_simple

  # Initialize segments
  segments = {}
  for u, v in G.edges:
    segments[G.edges[u, v]['id']] = [0, 0]

  ss_file = open(supersegments_file, 'w+')
  ssid = 0

  # Generate traffic and collect (super)segment data for each trip
  with open(trips_file, 'r') as f:
    i = 0
    for line in f.readlines():
      if len(line) <= 1:
        break

      tokens = line.split(' ')
      src, dst = int(tokens[0]), int(tokens[1])
      raw_supersegments = traffic_fn(G, src, dst, segments)

      # Write supersegments to file
      for eta, length, segment_ids in raw_supersegments:
        ss_data = format_supersegment(ssid, eta, length, segment_ids)
        ss_file.write(json.dumps(ss_data) + '\n')
        ssid += 1

      i += 1
      if i % 100 == 0:
        print(i)
      if i == 1000:
        break


  # Write segments to file
  with open(segments_file, 'w+') as f:
    for k, v in sorted(segments.items(), key=lambda x: x[0]):
      segment_data = format_segment(k, v)
      f.write(json.dumps(segment_data) + '\n')
