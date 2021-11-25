import json
import networkx as nx
from typing import Dict, List

from preprocess import load_graph

def load_segments(file_name: str) -> Dict:
  segment_etas = {}

  num_nonzero = 0
  with open(file_name, 'r') as f:
    for line in f.readlines():
      if len(line) <= 1:
        break
      segment_data = json.loads(line)
      segment_etas[segment_data['id']] = segment_data['average_eta']
      if segment_data['average_eta'] != 0:
        num_nonzero += 1

  print("Percent segments used:", num_nonzero / len(segment_etas))
  return segment_etas

def load_supersegments(file_name: str, hash_segments: bool = True) -> List:
  """
  Loads supersegment data from file.
  If `hash_segments` is true, replace list of segment ids with its hash 
  (to save memory).

  Output: List of supersegment objects. Each object has the following fields:
          - Supersegment id, as identified by the hash of its segments
          - Eta
          - Original length of supersegment
          - List of segment ids, or its hash
  """

  supersegments = []

  with open(file_name, 'r') as f:
    for line in f.readlines():
      if len(line) <= 1:
        break
      data = json.loads(line)
      if hash_segments:
        data['segment_ids'] = hash(tuple(data['segment_ids']))
      supersegments.append(data)

  # print(supersegments[0])
  return supersegments

if __name__ == '__main__':
  G = load_graph('SF.cnode', 'SF.cedge')
  print(f"SF road network has {len(G.nodes())} nodes and {len(G.edges)} edges")

  segment_etas = load_segments('segments.json')
  supersegments = load_supersegments('supersegments.json')