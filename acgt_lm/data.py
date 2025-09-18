import numpy as np
import pandas as pd

from misc import *

def sample_dna_chain(initial, transition, length, n_sequences=1, seed=42):
    rng = np.random.default_rng(seed)

    def generate_one():
        seq = [rng.choice(NUCLEOTIDES, p=initial)]
        for _ in range(length - 1):
            current = seq[-1]
            next_nuc = rng.choice(NUCLEOTIDES, p=transition[NUC_INDEX[current]])
            seq.append(next_nuc)

        return "".join(seq)

    sequences = [generate_one() for _ in range(n_sequences)]
    df = pd.DataFrame({"sequence": sequences})

    return df

if __name__ == "__main__":
  initial = [0.25, 0.25, 0.25, 0.25]
  transition = [
      [0.1, 0.4, 0.4, 0.1],      # A -> [A, C, G, T]
      [0.3, 0.2, 0.3, 0.2],      # C -> [A, C, G, T]
      [0.25, 0.25, 0.25, 0.25],  # G -> [A, C, G, T]
      [0.5, 0.2, 0.2, 0.1]       # T -> [A, C, G, T]
  ]

  df = sample_dna_chain(initial, transition, length=100, n_sequences=100)
  df.to_pickle('data.pkl')