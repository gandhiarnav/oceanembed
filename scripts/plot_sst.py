import xarray as xr
import matplotlib.pyplot as plt

file = "NASA_sst_data/sst_2020-01-01_0.25deg.nc"

sst = xr.open_dataarray(file)

sst = sst.squeeze()

plt.figure(figsize=(10, 6))
sst.plot()
plt.title("CMC SST — North Indian Ocean — 2020-01-01")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig("NASA_sst_data/sst_2020-01-01.png", dpi=150, bbox_inches="tight")
print("Saved plot.")
