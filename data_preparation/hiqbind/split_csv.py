#!/usr/bin/env python
"""Split a CSV into N chunks by unique PDBID."""
import pandas as pd
import sys, os

def split_csv(input_file, n_chunks, output_dir):
    df = pd.read_csv(input_file, dtype=str)
    pdbids = df['PDBID'].unique()
    chunk_size = max(1, len(pdbids) // n_chunks + (1 if len(pdbids) % n_chunks else 0))

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_file))[0]

    created = 0
    for i in range(n_chunks):
        ids = pdbids[i * chunk_size : (i + 1) * chunk_size]
        if len(ids) == 0:
            break
        chunk = df[df['PDBID'].isin(ids)]
        out = os.path.join(output_dir, f"{base}_chunk_{i+1}.csv")
        chunk.to_csv(out, index=False)
        created += 1
        print(f"  {out}: {len(ids)} PDBIDs, {len(chunk)} rows")
    return created

if __name__ == "__main__":
    n = split_csv(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    print(f"Created {n} chunks")
