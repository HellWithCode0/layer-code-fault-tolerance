# Numerical threshold estimate

## Scope

Finite-size **circuit-level crossing estimate** for the continuous algebraic
`base → gauged-strip → base` code-switch under an SI1000-like parameterization on the growing HGP/layer-code substrate,
with exact graphlike Stim DEM + MWPM.

**Not** a proof of the asymptotic committed-MW threshold in the FT theorem,
**not** a CAD-complete Williamson–Baspin T3/T4 DEM, and **not** an efficient-decoder theorem.

Paper wording: *finite-size crossing estimate for the algebraic code-switch model under an SI1000-like parameterization*.

## Headline estimate

- Median over G=d,2d,3d Jeffreys medians: **p_th^FS ≈ 0.004431**
- Envelope of per-duration Jeffreys 95% crossing bands: **[0.004265, 0.004522]**
- Total Monte Carlo shots (all families): **4,534,000**

## G=1d

- sizes: [3, 5, 7, 9, 11]
- noiseless all-zero: True (5 points)
- monotone p: [0.0015, 0.0025, 0.003, 0.00325, 0.0035, 0.00375, 0.004, 0.004125, 0.00425]
- Wilson-separated monotone p: [0.0015, 0.0025, 0.003, 0.00325, 0.0035, 0.00375]
- linear Lmin–Lmax crossing: 0.00447062
- **Jeffreys median: 0.00447072**
- Jeffreys 95%: [0.00442613, 0.00452183] (valid frac 1.000)
- shots: 1,548,000
- log p_L vs log L slopes: {'0.0015': -4.8352801313093625, '0.0025': -2.4823976788951314, '0.003': -1.662828630327854, '0.00325': -1.2835584099294204, '0.0035': -0.9277750670180467, '0.00375': -0.6194949354577635, '0.004': -0.37990971336598345, '0.004125': -0.27633011777156125, '0.00425': -0.2015033909322132, '0.004375': -0.06124858920116627, '0.0045': 0.020642375345790373, '0.004625': 0.09131613545143925, '0.00475': 0.16868951912609811, '0.005': 0.25844091962354304, '0.00525': 0.38719779659504255, '0.0055': 0.4101849189682387}

## G=2d

- sizes: [3, 5, 7, 9, 11]
- noiseless all-zero: True (5 points)
- monotone p: [0.0015, 0.0025, 0.003, 0.00325, 0.0035, 0.00375, 0.004, 0.004125]
- Wilson-separated monotone p: [0.0025, 0.003, 0.00325, 0.0035, 0.00375]
- linear Lmin–Lmax crossing: 0.00443056
- **Jeffreys median: 0.00443052**
- Jeffreys 95%: [0.00439183, 0.0044657] (valid frac 1.000)
- shots: 1,512,000
- log p_L vs log L slopes: {'0.0015': -4.415830125244422, '0.0025': -2.4234380965751896, '0.003': -1.6262780599771234, '0.00325': -1.1907832671527476, '0.0035': -0.902448207769728, '0.00375': -0.6019766252116937, '0.004': -0.32470395314271827, '0.004125': -0.23072146232643262, '0.00425': -0.14254205331130507, '0.004375': -0.04163534395507327, '0.0045': 0.0480301849737822, '0.004625': 0.12396866799572842, '0.00475': 0.17108615857387596, '0.005': 0.2613114191243674, '0.00525': 0.37433258116833673, '0.0055': 0.4065306889650745}

## G=3d

- sizes: [3, 5, 7, 9, 11]
- noiseless all-zero: True (5 points)
- monotone p: [0.0015, 0.0025, 0.003, 0.00325, 0.0035, 0.00375, 0.004, 0.004125]
- Wilson-separated monotone p: [0.0015, 0.0025, 0.003, 0.00325, 0.0035, 0.00375]
- linear Lmin–Lmax crossing: 0.00432751
- **Jeffreys median: 0.00432756**
- Jeffreys 95%: [0.00426489, 0.00439253] (valid frac 1.000)
- shots: 1,474,000
- log p_L vs log L slopes: {'0.0015': -4.8355839218341234, '0.0025': -2.4016314598756727, '0.003': -1.4943886676767235, '0.00325': -1.182064408026219, '0.0035': -0.8720722826021025, '0.00375': -0.5096371987697114, '0.004': -0.29923678935877923, '0.004125': -0.1643282887971299, '0.00425': -0.03627510395880267, '0.004375': 0.0295723057119244, '0.0045': 0.08511243594134692, '0.004625': 0.13964125789075965, '0.00475': 0.20301249781261851, '0.005': 0.2979140397042991, '0.00525': 0.38167527931614065, '0.0055': 0.38406966607654874}
