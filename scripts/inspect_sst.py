import xarray as xr
import numpy as np
file = "NASA_sst_data/20200101120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc"

ds = xr.open_dataset(file)

print("\n===== DATASET =====")
print(ds)

print("\n===== DIMENSIONS =====")
print(ds.dims)

print("\n===== COORDINATES =====")
print(ds.coords)

print("\n===== VARIABLES =====")
print(ds.data_vars)

print("\n===== SST INFO =====")
sst = ds["analysed_sst"]

# Extract North Indian Ocean
sst_region = sst.sel(
    lat=slice(5, 30),
    lon=slice(45, 105)
)

print("\n===== NORTH INDIAN OCEAN SST =====")
print(sst_region)
print("Shape:", sst_region.shape)


sst_celsius = sst_region - 273.15
sst_celsius.attrs["units"] = "degree_Celsius"

print("\n===== SST IN CELSIUS =====")
print(sst_celsius)
print("Min:", float(sst_celsius.min()))
print("Max:", float(sst_celsius.max()))
print("Mean:", float(sst_celsius.mean()))

mask = ds["mask"]

mask_region = mask.sel(
    lat=slice(5, 30),
    lon=slice(45, 105)
)

print("\n===== MASK VALUES =====")
print(mask_region.values.min())
print(mask_region.values.max())
print("Unique values:", np.unique(mask_region.values))



print("\n===== MASK INFO =====")
print(mask.attrs)


# Target SIH grid
target_lat = np.arange(5, 30.0001, 0.25)
target_lon = np.arange(45, 105.0001, 0.25)

# Regrid SST from 0.1° → 0.25°
sst_regridded = sst_celsius.interp(
    lat=target_lat,
    lon=target_lon,
    method="linear"
)

print("\n===== REGRIDDED SST =====")
print(sst_regridded)
print("Shape:", sst_regridded.shape)
print("Lat count:", len(target_lat))
print("Lon count:", len(target_lon))


# Crop mask to North Indian Ocean
mask_region = mask.sel(
    lat=slice(5, 30),
    lon=slice(45, 105)
)

# Regrid mask to the 0.25° target grid
mask_regridded = mask_region.interp(
    lat=target_lat,
    lon=target_lon,
    method="nearest"
)

# Ocean = 1, Land = 0
ocean_mask = mask_regridded == 1

# Remove land from SST
sst_final = sst_regridded.where(ocean_mask)

print("\n===== FINAL MASKED SST =====")
print(sst_final)
print("Shape:", sst_final.shape)
print("NaN count:", int(sst_final.isnull().sum()))
print("Min:", float(sst_final.min()))
print("Max:", float(sst_final.max()))

print(sst)

print("\nAttributes:")
print(sst.attrs)

# Save processed SST
sst_final.to_netcdf(
    "NASA_sst_data/sst_2020-01-01_0.25deg.nc"
)

print("\nSaved processed SST.")