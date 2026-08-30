import numpy as np

print('=' * 60)
print('       OCEANEMBED DATASET VERIFICATION')
print('=' * 60)

# Load files
inputs  = np.load('data/dummy/dummy_inputs.npy')
targets = np.load('data/dummy/dummy_targets.npy')
dates   = np.load('data/dummy/dummy_dates.npy', allow_pickle=True)

# 1. SHAPES & DTYPES
print('\n[1] SHAPES & DTYPES:')
print(f'  dummy_inputs.npy  : shape={inputs.shape},  dtype={inputs.dtype}')
print(f'  dummy_targets.npy : shape={targets.shape}, dtype={targets.dtype}')
print(f'  dummy_dates.npy   : shape={dates.shape}')

expected_inputs  = (365, 7, 101, 241)
expected_targets = (365, 15, 101, 241)
print(f'  inputs  shape OK  : {inputs.shape  == expected_inputs}  (expected {expected_inputs})')
print(f'  targets shape OK  : {targets.shape == expected_targets}  (expected {expected_targets})')
print(f'  inputs  dtype OK  : {inputs.dtype  == np.float32}')
print(f'  targets dtype OK  : {targets.dtype == np.float32}')

# 2. SPATIAL GRID
print('\n[2] SPATIAL GRID:')
print(f'  Lat points : {inputs.shape[2]}  (expected 101 -> 5N to 30N at 0.25 deg)')
print(f'  Lon points : {inputs.shape[3]}  (expected 241 -> 45E to 105E at 0.25 deg)')

# 3. TEMPORAL
print('\n[3] TEMPORAL:')
print(f'  Days       : {len(dates)}  (expected 365)')
print(f'  Date range : {dates[0]} to {dates[-1]}')

# 4. NaN CHECK
print('\n[4] NaN CHECK:')
print(f'  NaNs in inputs  : {np.isnan(inputs).sum()}')
print(f'  NaNs in targets : {np.isnan(targets).sum()}')

# 5. INPUT CHANNEL SUMMARY
channel_names = ['SST', 'SSS', 'SSH/SLA', 'U_curr', 'V_curr', 'U_wind', 'V_wind']
print('\n[5] INPUT CHANNELS:')
print(f'  {"Idx":<5}{"Channel":<12}{"Min":>10}{"Max":>10}{"Mean":>10}{"Std":>10}')
print('  ' + '-'*55)
for i, name in enumerate(channel_names):
    ch = inputs[:, i, :, :]
    print(f'  {i:<5}{name:<12}{ch.min():>10.4f}{ch.max():>10.4f}{ch.mean():>10.4f}{ch.std():>10.4f}')

# 6. TARGET DEPTH LEVELS
depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
print('\n[6] TARGET DEPTH TEMPERATURES:')
print(f'  {"Idx":<5}{"Depth_m":<10}{"Min":>10}{"Max":>10}{"Mean":>10}{"Std":>10}')
print('  ' + '-'*55)
for i, d in enumerate(depths):
    lv = targets[:, i, :, :]
    print(f'  {i:<5}{d:<10}{lv.min():>10.4f}{lv.max():>10.4f}{lv.mean():>10.4f}{lv.std():>10.4f}')

print('\n' + '=' * 60)
print('OVERALL STATUS:')
all_ok = (
    inputs.shape  == expected_inputs  and
    targets.shape == expected_targets and
    inputs.dtype  == np.float32       and
    targets.dtype == np.float32       and
    np.isnan(inputs).sum()  == 0      and
    np.isnan(targets).sum() == 0      and
    len(dates) == 365
)
status = 'OK' if all_ok else 'FAIL'
print(f'  [{status}] Data matches expected SIH-26066 format: {all_ok}')
print('=' * 60)
