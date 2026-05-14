#!/usr/bin/env python3
"""Preprocess references.json.gz into compact binary with IVF index.
Usage: python3 preprocess_ivf.py [n_clusters] [input_path] [output_path]
Defaults: n_clusters=2048, input=resources/references.json.gz, output=resources/references.bin
"""
import gzip
import json
import struct
import sys
import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def main():
    n_clusters = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    input_path = sys.argv[2] if len(sys.argv) > 2 else "resources/references.json.gz"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "resources/references.bin"

    t0 = time.time()
    print(f"Loading {input_path}...")
    with gzip.open(input_path, 'rt') as f:
        refs = json.load(f)

    vectors = np.array([r['vector'] for r in refs], dtype=np.float32)
    labels = np.array([1 if r['label'] == 'fraud' else 0 for r in refs], dtype=np.uint8)
    count, dim = vectors.shape
    print(f"Loaded {count} vectors, dim={dim}, clusters={n_clusters}")

    print(f"Building IVF (MiniBatchKMeans {n_clusters} clusters)...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=16384, n_init=1, max_iter=200, random_state=42)
    kmeans.fit(vectors)
    centroids = kmeans.cluster_centers_.astype(np.float32)
    assignments = kmeans.predict(vectors)
    print(f"  K-means done in {time.time()-t0:.1f}s")

    clusters = []
    for c in range(n_clusters):
        mask = assignments == c
        clusters.append(np.where(mask)[0].astype(np.int32))
        if c % 512 == 0:
            print(f"  Cluster {c}: {len(clusters[-1])} vectors")

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<III', count, dim, n_clusters))
        f.write(centroids.tobytes())
        for c in range(n_clusters):
            indices = clusters[c]
            f.write(struct.pack('<I', len(indices)))
            f.write(indices.tobytes())
        f.write(vectors.tobytes())
        f.write(labels.tobytes())

    file_size = len(vectors)*4 + len(labels) + 12 + n_clusters*dim*4 + sum(len(c)*4 + 4 for c in clusters)
    print(f"Saved: {output_path} ({file_size/(1024*1024):.1f}MB) in {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
