from preprocess import load_graph, generate_trips, generate_traffic_for_trips

def main():
  # Load input files into graph
  G = load_graph(nodes_file='SF.cnode', edges_file='SF.cedge')
  print(f"SF road network has {len(G.nodes())} nodes and {len(G.edges)} edges")

  # Generate random src/dst pairs for trips
  # trips_file = generate_trips(G, 1000, strategy='boxselect')
  trips_file = 'trips_boxselect.txt'

  # For each trip:
  # 1. Add random traffic
  # 2. Calculate shortest path from src/dst
  # 3. Generate data for segments and supersegments
  #    - Segments: average ETA across all trips, segment length
  #    - Supersegments: raw ETA, supersegment length, segment id's 
  segments_file = 'segments.json'
  supersegments_file = 'supersegments.json'
  generate_traffic_for_trips(G, trips_file, segments_file, supersegments_file)

if __name__ == '__main__':
  main()