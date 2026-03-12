# Scientific Validation of Preprocessing Decisions

Every claim in this document is tied to a specific source with PMID/DOI. Claims that could not be verified are explicitly marked **[UNVERIFIED]**. Where a source was read but does not support a previously assumed claim, this is noted.

---

## 1. TTL Synchronization

### Chandni's approach
Multi-pulse TTL barcode matching between pyPhotometry and Open Ephys, linear regression on matched pulse times for clock drift correction, PCHIP interpolation to upsample photometry to ECoG rate, trim unmatched ends.

### Assessment

| Component | Source | What the source actually says |
|---|---|---|
| Multi-pulse sync with linear regression for drift | Dagher & Ishiyama (2023), STAR Protocols 4(2):102306. PMID 37178112. DOI 10.1016/j.xpro.2023.102306 | Describes a protocol for precise signal synchronization of electrophysiology, videography, and audio using a custom pulse generator. Covers signal analysis, temporal alignment, and duration normalization. |
| Multi-modal synchronization software | Klumpp et al. (2025), Nature Communications 16:708. PMID 39814800. DOI 10.1038/s41467-025-56081-9 | Presents Syntalos software for precise synchronization of simultaneous multi-modal data acquisition and closed-loop interventions. |
| Simultaneous electrophysiology + fiber photometry | Patel et al. (2020), Frontiers in Neuroscience 14:148. PMID 32153363. DOI 10.3389/fnins.2020.00148 | Describes methods for simultaneous electrophysiology and fiber photometry in freely behaving mice. |
| pyControl Rsync synchronization | pyControl docs: https://pycontrol.readthedocs.io/en/latest/user-guide/synchronisation/ | **[NOT VERIFIED — URL not fetched this session]** |
| Open Ephys sync documentation | Open Ephys docs: https://open-ephys.github.io/gui-docs/Tutorials/Data-Synchronization.html | **[NOT VERIFIED — URL not fetched this session]** |
| PCHIP interpolation properties | Mathematical property of the algorithm | PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) preserves monotonicity and produces smooth interpolations without the ringing artifacts of sinc interpolation. This is a mathematical property, not an empirical claim. |

### Drift magnitudes
**[UNVERIFIED]** No published documentation was found quantifying clock drift for pyPhotometry (STM32) or Open Ephys (Intan) systems. Typical crystal oscillators have tolerances of 10-100 ppm, which would produce tens to hundreds of milliseconds of misalignment per hour, but specific values for these systems are not documented in any source we could verify.

---

## 2. ECoG Filtering

### Chandni's approach
4th order Butterworth bandpass 1-70 Hz + 60 Hz notch (Q=30), zero-phase sosfiltfilt.

### Assessment

| Parameter | Source | What the source actually says |
|---|---|---|
| Filter design guidelines (FIR and IIR) | Widmann et al. (2015), J. Neuroscience Methods 250:34-46. PMID 25128257. DOI 10.1016/j.jneumeth.2014.08.002 | Provides practical guidelines for evaluation of filter responses and selection of filter types (FIR/IIR) and parameters for electrophysiological data. Covers both FIR and IIR — does NOT specifically recommend one over the other. The paper discusses strategies for recognizing adverse filter effects and artifacts. |
| General filter guidance for neuroscience | de Cheveigné & Nelken (2019), Neuron 102(2):280-293. PMID 30998899. DOI 10.1016/j.neuron.2019.02.039 | Review on when, why, and how (not) to use filters. Recommends reporting filter characteristics with sufficient details. |
| False ripples from filtering sharp transients | Bénar et al. (2010), Clinical Neurophysiology 121(3):301-310. PMID 19955019. DOI 10.1016/j.clinph.2009.10.019 | Demonstrates that "filtering of sharp transients (e.g., epileptic spikes or artefacts) or signals with harmonics can result in 'false' ripples." Recommends high-pass filtering "should be performed with great care" and verified against original traces. Primarily concerns high-frequency oscillation detection (>80 Hz), not standard 1 Hz high-pass. |

### What we can and cannot say about these parameters

**4th order Butterworth:** Widely used in rodent ECoG literature. None of the sources above specifically endorse or reject this choice. Widmann 2015 covers both FIR and IIR without recommending one universally.

**1-70 Hz bandpass:** Standard range for interictal spike detection. **[No specific source verified for this exact range.]**

**60 Hz notch Q=30:** Standard practice for power line removal in US recordings. **[No specific source verified for Q=30.]**

**Zero-phase (sosfiltfilt):** Preserves event timing by filtering forward and backward. Appropriate for offline analysis. This is a well-established signal processing principle.

**SOS form vs ba-form:** SOS (second-order sections) avoids numerical instability of high-order polynomial transfer functions. The 4th-order bandpass becomes 8th order in ba-form, which is in the range where floating-point precision degrades. **[SciPy documentation recommends sosfiltfilt — not verified this session but can be checked at scipy.org]**

### Notch before or after bandpass?
Both are linear time-invariant operations. Convolution is commutative, so order does not affect the result. This is a mathematical property, not an empirical claim.

---

## 3. Fiber Photometry: Isosbestic Correction

### Three strategies in Spec

**Strategy A — Chandni's:** Gaussian smoothing σ=75 → simple subtraction: `ΔF/F = (GCaMP - Iso) / Iso`
**Strategy B — Meiling's:** Butterworth low-pass → biexponential detrend → OLS regression → ΔF/F
**Strategy C — IRLS:** Biexponential detrend → IRLS robust regression → ΔF/F

### The definitive OLS vs IRLS comparison

**Source:** Keevers & Jean-Richard-dit-Bressel (2025). "Obtaining artifact-corrected signals in fiber photometry via isosbestic signals, robust regression, and dF/F calculations." Neurophotonics 12(2):025003. PMID 40166421. PMC11957252. DOI 10.1117/1.NPh.12.2.025003

**What the paper actually says (verified from full text):**

1. **Methods compared:** OLS vs IRLS regression for fitting isosbestic to signal channel. Does **NOT** test simple subtraction (Strategy A).

2. **IRLS is superior:** "IRLS surpassed OLS regression for fitting isosbestic control signals to experimental signals." "IRLS regression produced more accurate artifact-corrected signals than OLS regression, with this effect scaling with the tuning constant."

3. **Why OLS fails:** "OLS regression assumes that all datapoints in the experimental signal should be used to fit the isosbestic signal. This is unsuitable for fiber photometry data because the experimental signal is, by design, expected to diverge from the isosbestic signal." OLS causes "overfitting of the isosbestic signal to the neural dynamic component of the experimental signal, while underfitting to the target artifactual component."

4. **OLS artifact description:** "OLS-based transients were downshifted relative to IRLS-based transients" and "mean waveforms (both dF and dF/F) were downshifted relative to the true signal." The paper does NOT use the phrase "spurious inhibition."

5. **IRLS implementation:** Uses MATLAB's `robustfit` with Tukey's bisquare weighting function. Tested tuning constants c=4.685 (MATLAB default), c=3, and c=1.4. "More aggressive downweighting reliably resulted in better accuracy."

6. **Low-pass filter:** 3 Hz low-pass filter. "Applying a 3 Hz low-pass filter to experimental and isosbestic signals resulted in more accurate artifact-corrected signals."

7. **Normalization:** dF/F recommended over dF. "Baseline normalization via dF/F calculation outperformed the dF calculation during event periods."

### Python equivalent of Keevers' MATLAB implementation
Keevers uses MATLAB `robustfit` with Tukey's bisquare (c=1.4). A Python equivalent (NOT from the paper):
```python
import statsmodels.api as sm
rlm_model = sm.RLM(signal_channel, sm.add_constant(isosbestic_channel),
                    M=sm.robust.norms.TukeyBiweight(c=1.4))
rlm_result = rlm_model.fit()
fitted_isosbestic = rlm_result.fittedvalues
```
Note: `statsmodels` default Tukey c=4.685. Keevers found c=1.4 (most aggressive tested) performed best.

### What we know about Strategy A (simple subtraction)
No published paper was found that tests simple subtraction (`(signal - iso) / iso` without regression) against regression-based methods. Keevers 2025 only tests OLS vs IRLS. The field moved directly to regression-based methods starting with Lerner et al. (2015).

**First-principles concern:** Simple subtraction assumes the isosbestic channel has the same scale as the signal channel's artifact component. Without a fitted scaling coefficient, this assumption is unlikely to hold exactly. However, no published evidence was found that experimentally validates or invalidates this approach.

### Related fiber photometry references (verified metadata only)

| Paper | PMID | DOI | Relevance |
|---|---|---|---|
| Lerner et al. (2015). "Intact-Brain Analyses Reveal Distinct Information Carried by SNc Dopamine Subcircuits." Cell 162:635-647. | 26232229 | 10.1016/j.cell.2015.07.014 | Established regression-based isosbestic correction for fiber photometry |
| Simpson, Akam et al. (2024). "Lights, fiber, action! A primer on in vivo fiber photometry." Neuron 112(5):718-739. | 38103545 | 10.1016/j.neuron.2023.11.016 | Comprehensive review; says 2-10 Hz low-pass typical for GCaMP6f/dLight1; notes subtraction vs division for bleaching is "not systematically characterized" |
| Sherathiya et al. (2021). "GuPPy, a Python toolbox for the analysis of fiber photometry data." Scientific Reports 11:24212. | 34930955 | 10.1038/s41598-021-03626-9 | Open-source analysis toolbox |
| Donka et al. (2025). "PASTa: An Open-Source Analysis and Signal Processing Toolbox." Current Protocols 5(7):e70161. | 40607617 | 10.1002/cpz1.70161 | Offers 9 background correction options; IRLS is one alternative, NOT the default. Default uses frequency-domain scaling. |
| Martianova, Aronson & Proulx (2019). "Multi-Fiber Photometry to Record Neural Activity in Freely-Moving Animals." JoVE (152):e60278. | 31680685 | 10.3791/60278 | Multi-fiber photometry protocol |
| Creamer et al. (2022). "Correcting motion induced fluorescence artifacts in two-channel neural imaging." PLOS Computational Biology 18(9):e1010421. | 36170268 | 10.1371/journal.pcbi.1010421 | Tests TMAC (Bayesian model) vs ratio method and others for C. elegans two-channel imaging (NOT fiber photometry). TMAC 20x more accurate. |

### What Simpson/Akam 2024 actually says about correction methods (verified from full text)
- Low-pass filter: "2-10 Hz is typically used for GCaMP6f and dLight1"
- Subtraction vs division for bleaching: "We are not aware of any systematic characterization of this and different studies use subtraction or division for photobleaching correction"
- Isosbestic correction assumption: Notes that proportional artifact size between channels "will in general not hold exactly"
- Does NOT recommend excluding initial recording period for baseline
- Does NOT specify recommended sampling rates

---

## 4. Gaussian Smoothing σ=75 — Cutoff Frequency

### The math (not a citation — standard DSP)
Gaussian smoothing is a low-pass filter. The -3 dB cutoff frequency is:
```
f_cutoff = 1 / (2π × σ_seconds)
where σ_seconds = σ_samples / sampling_rate
```

| Sampling Rate | σ=75 in seconds | Cutoff frequency |
|---|---|---|
| 1000 Hz | 0.075 s | 2.12 Hz |
| 130 Hz | 0.577 s | 0.28 Hz |
| 100 Hz | 0.750 s | 0.21 Hz |

### Assessment
At ~130 Hz (common for pyPhotometry after demodulation), σ=75 produces a 0.28 Hz cutoff, which would eliminate virtually all transient information. At 1000 Hz, the 2.12 Hz cutoff is in the range of Keevers' recommended 3 Hz low-pass.

**The actual pyPhotometry sampling rate in the Mattis lab must be verified to assess this.**

### What sources say about low-pass cutoff
- Keevers 2025 (verified): Uses **3 Hz** low-pass
- Simpson/Akam 2024 (verified): Says **"2-10 Hz is typically used for GCaMP6f and dLight1"**

---

## 5. Biexponential Fitting — Subtract vs. Divide

### What Keevers 2025 says (verified)
"Baseline normalization via dF/F calculation outperformed the dF calculation during event periods." Recommends dF/F.

### What Simpson/Akam 2024 says (verified)
"We are not aware of any systematic characterization of this and different studies use subtraction or division for photobleaching correction."

### Assessment
Keevers provides the only systematic comparison we found and recommends dF/F. Simpson/Akam note the lack of systematic characterization.

---

## 6. Transient Detection Parameters

### 6.1 Detection method

`find_peaks` with prominence-based detection is used in custom pipelines. **[No single definitive source was found endorsing this as "standard" — it is simply common practice.]**

**Alternative:** Neugornet, O'Donovan & Ortinski (2021). "Comparative Effects of Event Detection Methods on the Analysis and Interpretation of Ca2+ Imaging Data." Frontiers in Neuroscience 15:620869. PMID 33841076. DOI 10.3389/fnins.2021.620869.

What this paper actually says (verified from abstract): Wavelet ridgewalking "broadly outperformed dF/F₀-based methods for both neuron and astrocyte recordings." Traditional thresholding "produced spurious events and fragmented transients." **Note:** This is about calcium imaging (widefield), not fiber photometry specifically.

### 6.2 Thresholds used by published tools

| Source | What they actually use (verified) |
|---|---|
| Wallace et al. 2025 (PMID 40801083) | Z-Score Method: z=2.6 cutoff. Prominence Method: z=1.0 cutoff. **Does NOT mention findpeaks prominence=2.** |
| GuPPy / Sherathiya 2021 (PMID 34930955, verified from full text) | 2-step MAD process: (1) Filter out events > 2 MAD above median in a moving window (default 15s), (2) Count peaks > 3 MAD of the resultant trace. Authors note this is "somewhat arbitrary" but "generally select peaks that align with a human observer's judgement." |
| pMAT / Bruno 2021 (PMID 33385438) | **[Threshold details not verified from full text this session]** |
| FPmotion / Hong | **[Not verified — could not find PMID this session]** |
| **Chandni (Spec)** | Prominence ≥ 1 z-score |

### 6.3 Maximum peak width ≤ 8 seconds
**[No specific source verified for typical FWHM values of GCaMP or GRAB transients in vivo.]** 8 seconds is a permissive upper bound. Keep configurable.

### 6.4 HPF 0.01 Hz for drift removal
**[No specific source verified for or against this parameter.]** General signal processing concern: a high-pass filter will create undershoots after large transients (well-known IIR filter artifact). Lower cutoff (0.001 Hz) or alternative approaches (sliding median, polynomial detrending) would reduce this risk.

---

## 7. Z-Scoring and Normalization

### Source
Wallace et al. (2025). "Fiber Photometry Analysis of Spontaneous Dopamine Signals: The Z-Scored Data Are Not the Data." ACS Chemical Neuroscience 16(17):3239-3256. PMID 40801083. DOI 10.1021/acschemneuro.5c00078. Preprint: PMID 40060421.

### What the paper actually says (verified)

1. **Three methods compared:** Z-Score Method (z=2.6 cutoff), Manual Method (hand-adjusted baselines), Prominence Method (z=1.0 cutoff, then returns to preprocessed %ΔF/F for kinetics).

2. **Drugs tested:** SCH23390 (D1 antagonist), cocaine (uptake inhibitor), raclopride (D2 antagonist).

3. **Z-scoring failures:**
   - SCH23390: "Z-scoring failed to identify any changes, due to its amplification of noise when signals were diminished."
   - Cocaine: Z-score method showed cocaine "reduced signal amplitude but had no effect on signal width or slope" — other methods detected increased width.
   - Raclopride: "Z-scoring failed to identify any of the changes in dopamine release and uptake kinetics."

4. **Conclusion (direct quote):** "Analysis of spontaneous dopamine signals requires assessment of the %ΔF/F values, and the use of z-scoring is not appropriate."

5. **The Prominence Method** "combines z-scoring with prominence assessment to tag individual peaks and then returns to the preprocessed data for kinetic analysis." This matches Chandni's approach (z-score for detection, raw ΔF/F for measurement).

### What Simpson/Akam 2024 say about z-scoring (verified)
Z-scoring "will remove the influence of any factors that either scale the signal size" or affect baseline. However, this "may remove variation of experimental interest, such as changes in signal across learning over multiple sessions."

### Baseline period exclusion
**[UNVERIFIED]** The old document claimed Simpson/Akam 2024 and Sherathiya 2021 recommend excluding the first 2-5 minutes of recording. **Simpson/Akam 2024 does NOT make this recommendation** (verified from full text). Sherathiya 2021 was not checked for this claim. This recommendation has no verified source.

### Statistical framework
Loewinger et al. (2025). "A statistical framework for analysis of trial-level temporal dynamics in fiber photometry experiments." eLife 13:RP95802. PMID 40073228. DOI 10.7554/eLife.95802. **[Full text not read — metadata only verified.]**

---

## 8. Temperature: Calibration, Smoothing, and the GCaMP Heating Artifact

### 8.1 Linear calibration
`T(°C) = 0.0981 × V(mV) + 8.81` from the Spec. **[No source verified for this calibration equation — presumably from lab-specific calibration data.]** Linear calibration is appropriate if the probe has a linearization circuit upstream, which is typical for commercial temperature probes (Physitemp, Warner Instruments). **[This claim about commercial probes is general knowledge, not verified from a specific source.]**

### 8.2 300-sample moving average
Appropriateness depends on sampling rate. At 1 kHz → 0.3s window (fine). At 100 Hz → 3s window (introduces 1.5s lag). For a slow heating ramp, temperature error from lag is small, but temporal alignment with neural events may be affected.

### 8.3 GCaMP/GFP temperature sensitivity

**What we actually know from verified sources:**

| Source | What it actually says |
|---|---|
| dos Santos (2012). "Thermal effect on Aequorea green fluorescent protein anionic and neutral chromophore forms fluorescence." J. Fluorescence 22(1):151-154. PMID 21826424. DOI 10.1007/s10895-011-0941-0 | Studied GFP fluorescence from 20-75°C at pH 7. Neutral form (399 nm excitation) quenches faster than anionic form (476 nm excitation) with temperature increase. The two forms have **different temperature coefficients.** No specific %/°C values in abstract. |
| Leiderman et al. (2006). "Transition in the temperature-dependence of GFP fluorescence." Biophysical Journal 90(3):1009-1018. PMID 16284263. DOI 10.1529/biophysj.105.069393 | Studies GFP proton transfer dynamics from 87K to 310K (-186°C to +37°C). This is about **sub-physiological temperatures** and proton wire dynamics, NOT about fluorescence intensity changes at 37-43°C. Cannot be cited for a "%/°C" fluorescence decrease claim. |

**What we CANNOT say:** The old document claimed "~1-1.5% per °C decrease" at physiological temperatures. **No verified source was found for this quantitative claim.** GFP fluorescence does decrease with temperature (dos Santos 2012 confirms this qualitatively), but the specific rate at 37-43°C for GCaMP or GRAB sensors has not been quantified in any source we could verify.

### 8.4 Isosbestic correction for temperature
dos Santos 2012 confirms that the neutral (405 nm-relevant) and anionic (470 nm-relevant) GFP chromophore forms have different temperature coefficients. This means isosbestic-based correction will be imperfect for temperature-related artifacts, since the 405 nm and 470 nm channels will not be affected equally by temperature.

### 8.5 GRAB sensor temperature sensitivity
**[No source found.]** GRAB sensors use cpGFP (same fluorophore as GCaMP), so qualitatively similar temperature effects on fluorescence are expected. Additionally, temperature could affect GPCR binding affinity (Kd), which would change the sensor's response independent of fluorescence. **Both of these are first-principles reasoning, not verified claims.**

---

## Summary: What is verified vs. not

### Verified claims with sources
| Claim | Source |
|---|---|
| IRLS > OLS for isosbestic regression | Keevers 2025 (PMID 40166421) |
| OLS causes "downshifted" signals due to overfitting neural dynamics | Keevers 2025 (PMID 40166421) |
| Tukey bisquare c=1.4 most accurate (of tested values) | Keevers 2025 (PMID 40166421) |
| 3 Hz low-pass recommended | Keevers 2025 (PMID 40166421) |
| dF/F outperforms dF normalization | Keevers 2025 (PMID 40166421) |
| Z-scoring inappropriate for transient property measurement | Wallace 2025 (PMID 40801083) |
| Z-scoring failed to detect drug-induced kinetic changes | Wallace 2025 (PMID 40801083) |
| Prominence method (z-score detect → raw ΔF/F measure) works | Wallace 2025 (PMID 40801083) |
| 2-10 Hz low-pass typical for GCaMP6f/dLight1 | Simpson/Akam 2024 (PMID 38103545) |
| Subtraction vs division for bleaching not systematically characterized | Simpson/Akam 2024 (PMID 38103545) |
| High-pass filtering sharp transients creates "false ripples" | Bénar 2010 (PMID 19955019) |
| Neutral and anionic GFP forms have different temperature coefficients | dos Santos 2012 (PMID 21826424) |
| Wavelet ridgewalking outperformed dF/F thresholding in Ca2+ imaging | Neugornet 2021 (PMID 33841076) |
| GuPPy uses 2-step MAD: 2 MAD filter then 3 MAD threshold | Sherathiya 2021 (PMID 34930955) |
| PASTa offers 9 correction methods; IRLS is alternative, not default | Donka 2025 (PMID 40607617) |
| Creamer TMAC is for C. elegans two-channel imaging, not fiber photometry | Creamer 2022 (PMID 36170268) |


### Recommendations that remain sound despite sourcing issues
| Recommendation | Basis |
|---|---|
| Use IRLS (Strategy C) as default over OLS (Strategy B) | Keevers 2025 — strong evidence |
| Use z-ΔF/F for means, raw ΔF/F for transient properties | Wallace 2025 — strong evidence |
| Verify Gaussian σ=75 cutoff against actual sampling rate | Mathematical calculation |
| Use SOS form for filtering | Signal processing best practice (numerical stability) |
| Keep all detection parameters configurable | General good practice given lack of consensus |
| Temperature artifact is a concern for hyperthermia experiments | Qualitatively supported (dos Santos 2012), magnitude unquantified |
