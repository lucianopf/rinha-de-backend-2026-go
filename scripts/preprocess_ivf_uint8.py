#!/usr/bin/env python3
"""
Preprocess references.json.gz into uint8-quantized binary with IVF index.
Vector quantization: float32 → uint8 via per-dimension min/max scaling.
Stores min/max arrays in binary header for dequantization during search.

Usage: python3 preprocess_ivf_uint8.py [n_clusters] [input_path] [output_path]
Defaults: n_clusters=2048, input=resources/references.json.gz, output=resources/references.bin

Binary format:
  [0:4]   uint32 count (LE)
  [4:8]   uint32 dim (LE)
  [8:12]  uint32 nClusters (LE)
  [12:12+dim*4]   float32 mins[dim] (LE) — per-dimension minimum
  [12+dim*4:12+dim*8] float32 maxs[dim] (LE) — per-dimension maximum
  [12+dim*8:...] uint8 centroids[nClusters * dim]
  [...] cluster assignments (same as before)
  [...] uint8 vectors[count * dim]
  [...] uint8 labels[count] (0=legit, 1=fraud)
"""
import gzip
import json
import struct
import sys
import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def quantize(arr, mins, maxs):
    """Quantize float32 array to uint8 using per-dimension min/max.
    Returns uint8 array (same shape as input).
    """
    # arr shape: (N, dim)
    dim = arr.shape[1]
    out = np.zeros_like(arr, dtype=np.uint8)
    for d in range(dim):
        rng = maxs[d] - mins[d]
        if rng == 0:
            out[:, d] = 0
        else:
            clipped = np.clip(arr[:, d], mins[d], maxs[d])
            out[:, d] = np.round((clipped - mins[d]) / rng * 255).astype(np.uint8)
    return out


def main():
    n_clusters = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    input_path = sys.argv[2] if len(sys.argv) > 2 else "resources/references.json.gz"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "resources/references.bin"

    t0 = time.time()
    print(f"[uint8] Loading {input_path}...")
    with gzip.open(input_path, 'rt') as f:
        refs = json.load(f)

    vectors = np.array([r['vector'] for r in refs], dtype=np.float32)
    labels = np.array([1 if r['label'] == 'fraud' else 0 for r in refs], dtype=np.uint8)
    count, dim = vectors.shape
    print(f"[uint8] Loaded {count} vectors, dim={dim}, clusters={n_clusters}")

    # Compute per-dimension min/max from float32 data
    mins = vectors.min(axis=0).astype(np.float32)
    maxs = vectors.max(axis=0).astype(np.float32)
    # Add small epsilon to avoid zero range
    for d in range(dim):
        rng = maxs[d] - mins[d]
        if rng == 0:
            maxs[d] = mins[d] + 1.0
        elif rng < 1e-6:
            maxs[d] += 1e-6
    
    print(f"[uint8] Per-dim ranges: min={mins.min():.3f}..{mins.max():.3f}, max={maxs.min():.3f}..{maxs.max():.3f}")

    # Quantize vectors
    vectors_u8 = quantize(vectors, mins, maxs)
    
    # Run k-means on FLOAT32 vectors (more accurate clustering)
    print(f"[uint8] Building IVF (MiniBatchKMeans {n_clusters} on float32)...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=16384, n_init=1, max_iter=200, random_state=42)
    kmeans.fit(vectors)
    centroids_f32 = kmeans.cluster_centers_.astype(np.float32)
    assignments = kmeans.predict(vectors)
    print(f"[uint8] K-means done in {time.time()-t0:.1f}s")

    # Quantize centroids
    centroids_u8 = quantize(centroids_f32, mins, maxs)

    # Build cluster index lists (same as before)
    clusters = []
    for c in range(n_clusters):
        mask = assignments == c
        clusters.append(np.where(mask)[0].astype(np.int32))
        if c % 512 == 0:
            print(f"  Cluster {c}: {len(clusters[-1])} vectors")

    # Write binary
    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<III', count, dim, n_clusters))
        # Per-dimension mins and maxs (float32 LE)
        f.write(mins.tobytes())
        f.write(maxs.tobytes())
        # Quantized centroids (uint8)
        f.write(centroids_u8.tobytes())
        # Cluster assignments
        for c in range(n_clusters):
            indices = clusters[c]
            f.write(struct.pack('<I', len(indices)))
            f.write(indices.tobytes())
        # Quantized vectors (uint8)
        f.write(vectors_u8.tobytes())
        # Labels (uint8)
        f.write(labels.tobytes())

    # Calculate file sizes
    header_sz = 12 + dim * 8
    centroid_sz = n_clusters * dim * 1
    cluster_sz = sum(len(c) * 4 + 4 for c in clusters)
    vector_sz = count * dim * 1
    label_sz = count
    total_sz = header_sz + centroid_sz + cluster_sz + vector_sz + label_sz
    
    print(f"[uint8] Saved: {output_path} ({total_sz/(1024*1024):.1f}MB) in {time.time()-t0:.1f}s")
    print(f"[uint8] Breakdown: header={header_sz}B, centroids={centroid_sz/(1024):.1f}KB, "
          f"clusters={cluster_sz/(1024*1024):.1f}MB, vectors={vector_sz/(1024*1024):.1f}MB, labels={label_sz/(1024*1024):.1f}MB")


if __name__ == '__main__':
    main()
