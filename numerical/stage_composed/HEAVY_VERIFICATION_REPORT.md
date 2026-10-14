# Heavy Stim verification report

## Scope

finite high-statistics SI1000 falsification campaign for the paper's protected gauging and stage-composed architecture consequences; not an asymptotic proof, not a direct numerical measurement of Theorem VI.25 p_th, and not a continuous CAD-complete Williamson-Baspin T3/T4 detector model.

The headline primitive and HGP/layer-code-substrate experiments require exact graphlike Stim DEM construction. Any truncated-WB run is diagnostic and is excluded from headline crossing claims.

## Machine / campaign

- Python: `3.13.5 (main, Jul 15 2026, 20:25:40) [GCC 14.2.0]`
- Platform: `Linux-6.12.94+-x86_64-with-glibc2.41`
- CPU count seen: 8
- RAM seen: 15.637985229492188
- workers: 2
- preset: `heavy`
- total successful parameter points: 180
- total Monte Carlo shots: 3,955,000

## DEM exactness

- Approximate DEM headline rows: **0**.
- Headline primitive/HP rows used exact graphlike DEM construction.

## hp_cas — stress — R=1d

Sizes: `[3, 5, 7, 9, 11]`

Monotone suppression points (all consecutive sizes): `[0.0015, 0.0025, 0.0035, 0.0045]`

Wilson-separated monotone points: `[0.0015, 0.0025, 0.0035]`

Finite-size crossing diagnostic (smallest vs largest size):

- observed bracket: `[0.0045, 0.004666666666666666]`
- interpolated median: **0.00464434**
- diagnostic 95% posterior interval: [0.00461339, 0.00466533]
- valid posterior interpolation fraction: 0.761
- interpretation: Jeffreys-posterior finite-size interpolation diagnostic; NOT an asymptotic threshold CI

Point estimates:

- p=0.0015: S3=0.0209333, S5=0.004475, S7=0.000941667, S9=0.000133333, S11=1.66667e-05 MONO WILSON-SEPARATED
- p=0.0025: S3=0.0576, S5=0.0262, S7=0.0099, S9=0.00433333, S11=0.00167619 MONO WILSON-SEPARATED
- p=0.0035: S3=0.105333, S5=0.0701333, S7=0.0495333, S9=0.0332667, S11=0.0244 MONO WILSON-SEPARATED
- p=0.0045: S3=0.165667, S5=0.156, S7=0.1468, S9=0.1436, S11=0.1382 MONO
- p=0.004666667: S3=0.1726, S5=0.173, S7=0.167, S9=0.175, S11=0.175533
- p=0.004833333: S3=0.183067, S5=0.1816, S7=0.193067, S9=0.202, S11=0.217133
- p=0.005: S3=0.197333, S5=0.201667, S7=0.216067, S9=0.238067, S11=0.2586
- p=0.005166667: S3=0.203, S5=0.2264, S7=0.245933, S9=0.260067, S11=0.300467
- p=0.005333333: S3=0.217133, S5=0.2448, S7=0.267933, S9=0.309733, S11=0.348
- p=0.0055: S3=0.221667, S5=0.250667, S7=0.3022, S9=0.3444, S11=0.389533
- p=0.0065: S3=0.292, S5=0.370733, S7=0.477, S9=0.58, S11=0.6544

Effective log-error slopes vs size (negative = suppression):

- p=0.0015: slope=-0.865678
- p=0.0025: slope=-0.443373
- p=0.0035: slope=-0.183416
- p=0.0045: slope=-0.0222639
- p=0.004666667: slope=0.0022595
- p=0.004833333: slope=0.0223854
- p=0.005: slope=0.0353303
- p=0.005166667: slope=0.0461385
- p=0.005333333: slope=0.0589254
- p=0.0055: slope=0.0722532
- p=0.0065: slope=0.103066

## hp_cas — stress — R=2d

Sizes: `[3, 5, 7, 9, 11]`

Monotone suppression points (all consecutive sizes): `[0.0015, 0.0025, 0.0035, 0.0036666666666666666, 0.003833333333333333, 0.004, 0.004166666666666667, 0.004333333333333333]`

Wilson-separated monotone points: `[0.0015, 0.0025, 0.0035, 0.0036666666666666666, 0.003833333333333333, 0.004]`

Finite-size crossing diagnostic (smallest vs largest size):

- observed bracket: `[0.004333333333333333, 0.0045]`
- interpolated median: **0.00443339**
- diagnostic 95% posterior interval: [0.0044072, 0.00446578]
- valid posterior interpolation fraction: 1.000
- interpretation: Jeffreys-posterior finite-size interpolation diagnostic; NOT an asymptotic threshold CI

Point estimates:

- p=0.0015: S3=0.0354667, S5=0.0089, S7=0.00176, S9=0.000375, S11=8.33333e-05 MONO WILSON-SEPARATED
- p=0.0025: S3=0.0929333, S5=0.0451333, S7=0.0210667, S9=0.0101, S11=0.00348 MONO WILSON-SEPARATED
- p=0.0035: S3=0.1704, S5=0.133867, S7=0.0982, S9=0.071, S11=0.0504 MONO WILSON-SEPARATED
- p=0.003666667: S3=0.1836, S5=0.1506, S7=0.124133, S9=0.0954667, S11=0.0728 MONO WILSON-SEPARATED
- p=0.003833333: S3=0.1974, S5=0.1698, S7=0.1484, S9=0.122, S11=0.1022 MONO WILSON-SEPARATED
- p=0.004: S3=0.214067, S5=0.1942, S7=0.1712, S9=0.153333, S11=0.1358 MONO WILSON-SEPARATED
- p=0.004166667: S3=0.229533, S5=0.221, S7=0.207867, S9=0.190533, S11=0.175067 MONO
- p=0.004333333: S3=0.25, S5=0.248133, S7=0.240667, S9=0.237267, S11=0.224333 MONO
- p=0.0045: S3=0.259667, S5=0.266867, S7=0.273067, S9=0.279933, S11=0.276733
- p=0.0055: S3=0.352, S5=0.4296, S7=0.497533, S9=0.570267, S11=0.6282
- p=0.0065: S3=0.440867, S5=0.5652, S7=0.6856, S9=0.7708, S11=0.8234

Effective log-error slopes vs size (negative = suppression):

- p=0.0015: slope=-0.758489
- p=0.0025: slope=-0.402996
- p=0.0035: slope=-0.153466
- p=0.003666667: slope=-0.115263
- p=0.003833333: slope=-0.0823405
- p=0.004: slope=-0.0573129
- p=0.004166667: slope=-0.0344992
- p=0.004333333: slope=-0.01307
- p=0.0045: slope=0.00875459
- p=0.0055: slope=0.0720801
- p=0.0065: slope=0.0779781

## primitive — gauge — R=1d

Sizes: `[3, 5, 7, 9, 11]`

Monotone suppression points (all consecutive sizes): `[0.0015, 0.0025, 0.0035]`

Wilson-separated monotone points: `[]`

Finite-size crossing diagnostic (smallest vs largest size):

- observed bracket: `[0.004, 0.004166666666666667]`
- interpolated median: **0.00411902**
- diagnostic 95% posterior interval: [0.00404352, 0.00416332]
- valid posterior interpolation fraction: 0.607
- interpretation: Jeffreys-posterior finite-size interpolation diagnostic; NOT an asymptotic threshold CI

Point estimates:

- p=0.0015: S3=0.00391111, S5=0.00135652, S7=0.000375, S9=0.000175, S11=5.83333e-05 MONO
- p=0.0025: S3=0.0114667, S5=0.00623333, S7=0.0037, S9=0.00291667, S11=0.00121667 MONO
- p=0.0035: S3=0.0218667, S5=0.0197333, S7=0.0179333, S9=0.0130667, S11=0.0113333 MONO
- p=0.003666667: S3=0.0251333, S5=0.0228667, S7=0.0196667, S9=0.0202667, S11=0.0152
- p=0.003833333: S3=0.0270667, S5=0.0232667, S7=0.0222, S9=0.0244667, S11=0.0201333
- p=0.004: S3=0.0310667, S5=0.0302, S7=0.0274, S9=0.0296667, S11=0.0266667
- p=0.004166667: S3=0.0333333, S5=0.0342, S7=0.0355333, S9=0.0336, S11=0.0338667
- p=0.004333333: S3=0.0355333, S5=0.0369333, S7=0.0388667, S9=0.0414667, S11=0.0408667
- p=0.0045: S3=0.0385333, S5=0.0437333, S7=0.0475333, S9=0.0469333, S11=0.0493333
- p=0.0055: S3=0.0582667, S5=0.0730667, S7=0.0936, S9=0.109267, S11=0.1366
- p=0.0065: S3=0.0782667, S5=0.112667, S7=0.158133, S9=0.207467, S11=0.256067

Effective log-error slopes vs size (negative = suppression):

- p=0.0015: slope=-0.515301
- p=0.0025: slope=-0.262239
- p=0.0035: slope=-0.0861494
- p=0.003666667: slope=-0.0562292
- p=0.003833333: slope=-0.0270393
- p=0.004: slope=-0.0161443
- p=0.004166667: slope=0.000701652
- p=0.004333333: slope=0.019756
- p=0.0045: slope=0.028217
- p=0.0055: slope=0.105283
- p=0.0065: slope=0.149022

## Interpretation boundary

A stable finite-size crossing, negative size slopes below it, exact DEMs, and persistence under R=d -> R=2d are strong finite numerical evidence for the implemented consequences. They are **not** a numerical proof of the paper's asymptotic threshold theorem. In particular, the stage-composed stress CAS does not yet instantiate one continuous CAD-complete Williamson-Baspin T3/T4 spacetime detector circuit.
