import pandas as pd
import numpy as np

p = pd.read_csv("results/paired_gwl_support.csv", index_col=0)
d = p["difference_m"].dropna().to_numpy()

rng = np.random.default_rng(42)
boot = np.array([
    rng.choice(d, size=len(d), replace=True).mean()
    for _ in range(10000)
])

print(f"Paired wells: {len(d)}")
print(f"Mean difference: {d.mean():.3f} m")
print(f"95% bootstrap CI: [{np.percentile(boot, 2.5):.3f}, "
      f"{np.percentile(boot, 97.5):.3f}] m")
print(f"Median difference: {np.median(d):.3f} m")
