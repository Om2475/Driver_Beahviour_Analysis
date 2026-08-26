# Driver Behaviour Analysis: Complete Technical Breakdown

**Project Overview:** End-to-end ML pipeline for analyzing OBD-II vehicle telemetry data to understand driver behavior patterns, detect risky driving, and assign behavioral contexts.

**Key Metrics:**
- Windows analyzed: 78,380
- Trips analyzed: 81
- Mean risk score: 0.2542 (95th percentile: 0.5539)
- Embedding quality score: 0.7776
- Top risk feature: throttle_mean_scaled

---

## **NOTEBOOK 01: Data Cleaning and Preprocessing**

### Input
- Raw OBD-II CSV files from `data/raw/OBD-II-Dataset/` (multiple trip files)

### Processing Pipeline

#### 1. Column Normalization
- Converts column names to lowercase, replaces special characters with underscores
- Maps standardized sensor names:
  - `vehicle_speed_sensor_km_h` → `speed`
  - `engine_rpm_rpm` → `rpm`
  - `absolute_throttle_position` → `throttle`
  - `intake_manifold_absolute_pressure_kpa` → `map`
  - `air_flow_rate_from_mass_flow_sensor_g_s` → `maf`
  - `accelerator_pedal_position_d/e` → `pedal_d/e`
  - Temperature sensors → `coolant_temp`, `intake_temp`, `ambient_temp`

#### 2. Time Conversion & Validation
**Formula:**
$$t_{seconds} = hours \times 3600 + minutes \times 60 + seconds$$

**Normalized trip time:**
$$t_{normalized} = t_{current} - t_{min}$$

- Converts mixed time formats (HH:MM:SS or elapsed seconds) to numeric seconds
- Handles missing time values via forward/backward fill
- Creates per-trip normalized timestamps

#### 3. Temporal Quality Control
**Adaptive gap detection:**
$$threshold = median(\Delta t) \times 3$$

- Detects data gaps using adaptive threshold (3x median time delta)
- Flags gaps with `gap_flag` column (0/1)
- Removes duplicate timestamps within trips
- Validates strict monotonic ordering per trip

#### 4. Interpolation & Filling
- **Per-trip interpolation** with `limit=3` (max 3 consecutive NaN gaps)
- Forward/backward fill for remaining gaps
- **Sensor-wise** (not global) to preserve signal integrity
- Applied to all 9 sensor columns

#### 5. Outlier Clipping
Fixed realistic bounds:
- `speed`: [0, 200] km/h
- `rpm`: [0, 8000] RPM
- `throttle`: [0, 100] %
- `map`: [0, 300] kPa
- `coolant_temp`: [-40, 150] °C
- `intake_temp`, `ambient_temp`: [-40, 100] °C

#### 6. Final Schema & Validation
**Output columns:** `trip_id, time, gap_flag, speed, rpm, throttle, map, maf, pedal_d, coolant_temp, intake_temp, ambient_temp`

**Validation assertions:**
- Time is monotonically increasing per trip
- No duplicate (trip_id, time) pairs
- No missing values in sensor columns
- No infinite values

### Output Files
1. **`clean_obd_data.csv`**: Main cleaned dataset (no missing values, no infinities)
2. **`trip_summary.csv`**: Per-trip statistics (min/max time, mean/std speed, mean/std rpm)

---

## **NOTEBOOK 02: Exploratory Data Analysis**

### Input
- `data/processed/clean_obd_data.csv`

### Exploratory Analyses

#### 1. Basic Data Overview
- Shape, schema, data types inspection
- Missing value counts per column (should be zero after cleaning)
- Descriptive statistics per sensor

#### 2. Distribution Analysis
**Histograms** (60 bins) with density plots:
- Speed, RPM, Throttle, Pedal Position, MAP, MAF, Coolant Temperature
- Identifies multi-modal distributions (different driving regimes)

#### 3. Time-Series Inspection
- Single-trip trajectories for speed, rpm, throttle, MAP, coolant_temp
- Reveals nonstationary behavior, acceleration patterns, thermal dynamics

#### 4. Sensor Relationship Analysis
**Scatter plots:**
- Throttle vs. RPM (driver input vs. engine response)
- RPM vs. Speed (engine load vs. vehicle velocity)
- MAP vs. RPM (intake pressure vs. engine speed)

#### 5. Correlation Matrix
**Pearson correlation:**
$$r_{ij} = \frac{cov(X_i, X_j)}{\sigma_i \sigma_j}$$

- Full heatmap of sensor correlations
- Identifies multicollinear signals (e.g., RPM-Speed, Throttle-MAP)

#### 6. Trip-Level Aggregation
Per-trip statistics: mean/max speed, mean/max RPM, mean throttle, mean MAP, mean MAF
- Reveals inter-trip variability for behavior classification

#### 7. Outlier Analysis
- Boxplots for speed, RPM, throttle
- IQR-based outlier detection (1.5 × IQR rule)

### Output
- Matplotlib/Seaborn visualizations (displayed in notebook)
- Summary statistics for feature engineering decisions

---

## **NOTEBOOK 03: Feature Engineering**

### Input
- `data/processed/clean_obd_data.csv`

### Processing

#### Resampling (1Hz)
- Trip-wise aggregation to 1-second frequency (mean)
- Interpolate missing values with `limit=2`
- Forward/backward fill remaining gaps

#### Window Parameters
- **Window size**: 5 seconds (@ 1Hz = 5 samples)
- **Stride**: 2 seconds
- **Multi-scale context**: 10 seconds (for frequency-domain features)
- **Per-trip thresholds**: Computed once, applied across all windows

### Feature Families (~110+ features)

#### A. Core Statistical Features (per signal)
For each signal $x$ (speed, rpm, throttle, map, coolant_temp):

$$\text{mean} = \frac{1}{N}\sum_{i=1}^N x_i$$

$$\text{std} = \sqrt{\frac{1}{N}\sum_{i=1}^N (x_i - \bar{x})^2}$$

$$\text{skewness} = \frac{E[(x-\mu)^3]}{\sigma^3}$$

$$\text{kurtosis} = \frac{E[(x-\mu)^4]}{\sigma^4} - 3$$

$$\text{percentiles: } q25, median, q75$$

**Features per signal:** mean, std, min, max, median, q25, q75, skew, kurtosis = **9 features × 5 signals = 45 features**

#### B. Temporal Dynamics (~20 features)

**Acceleration & Jerk:**
$$\text{acceleration}[t] = \text{speed}[t] - \text{speed}[t-1]$$
$$\text{jerk}[t] = \text{acceleration}[t] - \text{acceleration}[t-1]$$

- `accel_mean, accel_std, accel_max, accel_min`
- `jerk_mean, jerk_std`
- `rpm_diff_mean, throttle_diff_mean`
- `speed_roll_var_mean` (3-sample rolling variance of speed)

**Rate Features:**
$$\text{zero_crossing_rate} = \frac{\text{# of sign changes}}{N-1}$$
$$\text{throttle_oscillation_rate} = \frac{\text{# direction changes in throttle derivative}}{N-1}$$

$$\text{positive_accel_ratio} = P(\text{accel} > 0)$$
$$\text{negative_accel_ratio} = P(\text{accel} < 0)$$

**Driving Intensity:**
$$\text{driving_intensity} = \text{mean}(\text{acceleration}^2)$$

#### C. Event Features (~8 features)
**Per-trip global thresholds:**
- `acc_high_global = Q90(\text{acceleration})`
- `acc_low_global = Q10(\text{acceleration})`
- `throttle_spike_global = Q90(|\Delta \text{throttle}|)`
- `speed_idle_global = Q20(\text{speed})`
- `rpm_idle_high_global = Q80(\text{rpm})`

**Event counts & ratios:**
- `count_hard_acceleration, ratio_hard_acceleration`
- `count_hard_braking, ratio_hard_braking`
- `count_throttle_spikes, ratio_throttle_spikes`
- `count_idle_high_rpm, ratio_idle_high_rpm`

#### D. Normalized Response Features (~8 features)
**Z-score within window:**
$$z_i = \frac{x_i - \bar{x}}{\sigma_x}$$

- `rpm_over_speed_norm = \frac{z_{rpm}}{z_{speed} + \epsilon}` → mean, std
- `speed_over_throttle_norm = \frac{z_{speed}}{z_{throttle} + \epsilon}` → mean, std
- `accel_over_throttle_norm = \frac{z_{accel}}{z_{throttle} + \epsilon}` → mean, std
- `rpm_over_throttle_norm = \frac{z_{rpm}}{z_{throttle} + \epsilon}` → mean, std

#### E. Correlation & Response-Delay Features (~4 features)
- `corr_throttle_speed = \text{Pearson}(\text{throttle}, \text{speed})`
- `corr_throttle_rpm = \text{Pearson}(\text{throttle}, \text{rpm})`
- `lag1_corr_throttle_speed = \text{Pearson}(\text{throttle}[:-1], \text{speed}[1:])`
- `lag1_corr_throttle_rpm = \text{Pearson}(\text{throttle}[:-1], \text{rpm}[1:])`

#### F. Engine & Context Features (~6 features)
- `map_q85 = Q85(\text{MAP})`
- `coolant_q15 = Q15(\text{coolant_temp})`
- `engine_load_proxy = \text{rpm} \times \text{map}`
- `pct_time_high_map = P(\text{MAP} > \text{map\_q85})`
- `pct_time_low_coolant = P(\text{coolant\_temp} < \text{coolant\_q15})`
- `engine_load_proxy_mean, engine_load_proxy_std`

#### G. Frequency-Domain & Entropy Features (~8 features)

**FFT Energy & Entropy** (on 10-sec context window):
$$\text{spectrum} = \text{FFT}(x - \text{mean}(x))$$
$$\text{power} = |\text{spectrum}|^2$$
$$\text{energy} = \frac{\sum \text{power}}{N}$$

**Entropy:**
$$\text{entropy} = -\sum p_k \log(p_k), \quad p_k = \frac{\text{power}_k}{\sum \text{power}}$$

- `speed_fft_energy, speed_fft_entropy`
- `throttle_fft_energy, throttle_fft_entropy`

**Histogram Entropy** (adaptive bins 3-15):
$$H = -\sum p_i \log(p_i + \epsilon)$$
- `speed_entropy, throttle_entropy, rpm_entropy`

#### H. Driver Micro-Behavior (~4 features)
$$\text{stable_accel_threshold} = Q30(|\text{accel}|)$$

- `throttle_smoothness = \frac{1}{1 + \text{std}(\text{throttle})}$
- `acceleration_aggressiveness = \text{std}(\text{accel})`
- `steady_speed_ratio = P(|\text{accel}| \leq \text{stable_accel_threshold})`
- `speed_variability_index = \frac{\text{std}(\text{speed})}{|\text{mean}(\text{speed})| + \epsilon}$

#### I. Multi-Signal Interaction & Consistency (~3 features)
- `cov_speed_rpm = \text{cov}(\text{speed}, \text{rpm})`
- `cov_throttle_acceleration = \text{cov}(\text{throttle}, \text{accel})`
- `consistency_index = \text{mean}([\text{std}(\text{speed}), \text{std}(\text{throttle}), \text{std}(\text{accel})])`

#### J. Multi-Scale (10-second context) (~8 features)
- `ms10_speed_mean, ms10_speed_std`
- `ms10_rpm_mean, ms10_rpm_std`
- `ms10_throttle_mean, ms10_throttle_std`
- `ms10_accel_std`

### Feature Scaling & Selection

#### Stability Filtering
- Drop constant-value features (nunique ≤ 1)
- Drop near-zero variance features (var ≤ 1e-8)

#### StandardScaler Normalization
$$X_{\text{scaled}} = \frac{X - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
- Fit on 80% train split only (prevent leakage)

#### Correlation Filtering (threshold: 0.9)
- Compute correlation matrix on scaled features
- Remove lower-variance feature from correlated pairs
- Preserve representation diversity (kinematic, powertrain, temporal)

### Output
- **`feature_dataset.csv`**: Window-level feature matrix
  - Rows: ~100K+ windows
  - Columns: metadata (4) + selected scaled features (~80-120 after filtering)
  - Each row = 1 sliding window (5 seconds @ 1Hz)

---

## **NOTEBOOK 04: Context Discovery** ← **PRIMARY FOCUS**

### Input
- `data/features/feature_dataset.csv` (~100K rows, ~90-120 features)

### Unsupervised Learning Pipeline

#### Phase 1: Data Preparation

**Missing Value Imputation**
- Trip-wise forward fill within `trip_id`
- Median imputation for remaining gaps
- SimpleImputer strategy: "median"

**Standardization**
$$X_{\text{scaled}} = \frac{X - \mu}{\sigma}$$
- Fit on all data

#### Phase 2: Representation Learning (4 embedding spaces)

**A. Raw Scaled Features (baseline)**
$$X_{\text{raw}} = X_{\text{scaled}}$$

**B. PCA (Linear Dimensionality Reduction)**
$$X_{\text{pca}} = X_{\text{scaled}} W_{\text{pca}}$$
- Components: retain **95% of variance** (typically 15-25 components)
- Saved: `models/context_discovery/pca.joblib`

**C. UMAP (Nonlinear Neighborhood Preservation)**
$$X_{\text{umap}} = \text{UMAP}(X_{\text{scaled}})$$
- `n_neighbors=15` (local structure)
- `min_dist=0.1` (global structure)
- `n_components=10`
- `metric='euclidean'`
- `random_state=42`

**D. Autoencoder Latent Space**
Architecture:
```
Input (d-dims) 
→ Dense(max(64, 2d), relu) 
→ BatchNorm 
→ Dense(max(32, d), relu)
→ Bottleneck/Latent(latent_dim, relu)
→ Dense(max(32, d), relu) 
→ Dense(max(64, 2d), relu) 
→ Output(d-dims, linear)
```
- Latent dimension: `max(8, min(32, d//2))` (typically 16-24)
- Loss: MSE reconstruction
- Trained on 20K balanced sample

#### Phase 3: Multi-Model Clustering Search

**Across:** 4 embeddings × 3 algorithms × multiple parameters = ~60 candidate models

**1. K-Means Clustering**
$$C = \arg\min_C \sum_{i=1}^K \sum_{x \in C_i} ||x - \mu_i||^2$$
- `n_clusters`: k ∈ [2, 10]
- Initialization: 'auto' (10 runs)
- Per-embedding: 9 models

**2. Gaussian Mixture Models (GMM)**
$$p(x) = \sum_{k=1}^K \pi_k N(x|\mu_k, \Sigma_k)$$
- `n_components`: k ∈ [2, 10]
- `covariance_type='diag'`
- `reg_covar=1e-6`
- Soft membership: $P(z_k|x) = \frac{\pi_k N(x|\mu_k, \Sigma_k)}{p(x)}$

**3. DBSCAN (Density-Based)**
$$\text{density}_i = \frac{|N_\epsilon(x_i)|}{V_\epsilon}$$
- `eps`: Data-driven via k-distance graph (percentiles 90-99)
- `min_samples`: max(5, d+1)
- Produces noise points (label = -1)

#### Phase 4: Clustering Quality Metrics

All metrics computed on validation set (~10K points):

**1. Silhouette Score** ([-1, 1], higher better)
$$S_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$
- $a_i$ = mean intra-cluster distance
- $b_i$ = mean nearest-cluster distance

**2. Davies-Bouldin Index** (lower better, min ≥ 0)
$$DB = \frac{1}{k}\sum_{i=1}^k \max_{j \neq i} \frac{S_i + S_j}{d(\mu_i, \mu_j)}$$
- Ratio of within-cluster to between-cluster separation

**3. Calinski-Harabasz Score** (higher better)
$$CH = \frac{\text{trace}(B_k)}{\text{trace}(W_k)} \cdot \frac{n-k}{k-1}$$
- $B_k$ = between-cluster scatter
- $W_k$ = within-cluster scatter

**4. Noise Ratio** (for DBSCAN)
$$\text{noise_ratio} = \frac{|\{x: \text{label}(x) = -1\}|}{N}$$

#### Phase 5: Model Selection

**Ranking formula:** Silhouette (primary), Davies-Bouldin (secondary), Calinski-Harabasz (tertiary), Noise Ratio (quaternary)

**Typical winner:**
- Embedding: UMAP or Autoencoder
- Model: KMeans or GMM
- Parameters: k ∈ [5, 8] contexts

#### Phase 6: Full-Dataset Label Assignment

Refit best model on full embedding:
```python
if model_type == "kmeans":
    labels = best_model.fit_predict(X_full_embedding)
elif model_type == "gmm":
    labels = best_model.fit_predict(X_full_embedding)
    soft_probabilities = best_model.predict_proba(X_full_embedding)
elif model_type == "dbscan":
    labels = best_model.fit_predict(X_full_embedding)
```

#### Phase 7: Confidence Scoring

**3-way ensemble confidence:**

**Component 1: Cluster Confidence**
- GMM: $P(z_k|x)$ (posterior probability)
- KMeans: $1 - \frac{d_i}{d_{\text{max}}}$ (inverse distance to centroid)
- DBSCAN: density-based

**Component 2: Reconstruction Confidence**
$$\text{recon_confidence} = 1 - \text{normalize}(\text{reconstruction_error})$$

**Component 3: Local Density Confidence**
$$\text{density_confidence} = \text{normalize}(\text{k-NN mean distance}, \text{invert}=True)$$

**Final Confidence:**
$$\text{context_confidence} = 0.5 \times \text{cluster\_conf} + 0.3 \times \text{recon\_conf} + 0.2 \times \text{density\_conf}$$
- Normalized to [0, 1]

#### Phase 8: Temporal Smoothing

**Rolling mode smoothing** (window=5) per trip:
$$\text{cluster_label_smoothed}[t] = \text{mode}(\text{cluster_label}[t-2:t+3])$$
- Reduces label noise while preserving transitions

#### Phase 9: Context Stability Metrics

**1. Transition Matrix** (row-normalized)
$$T_{ij} = \frac{\text{count(transition from } i \text{ to } j)}{N_i}$$

**2. Transition Score**
$$\text{transition_score} = 1 - \frac{\text{# cluster changes}}{\text{total windows}}$$

**3. Bootstrap Stability Score**
- Perturb embedding, refit model multiple times
- Measure agreement via Adjusted Rand Index

**4. Anomaly Detection**
$$\text{is_anomaly} = \text{reconstruction_error} \geq Q95(\text{reconstruction_error})$$

#### Phase 10: Cluster Profiling

Per cluster, compute z-score of feature means:
$$z_{ck} = \frac{\text{mean}_c(x_k) - \text{mean}_{\text{global}}(x_k)}{\text{std}(x_k)}$$
- Select top 12 features by |z-score|
- Report as "elevated" or "reduced"

### Output Files

1. **`context_labels.csv`**: Window-level labels
   - `trip_id, window_id, cluster_label`

2. **`final_features_with_context.csv`**: Enriched feature dataset
   - `cluster_label` (raw)
   - `cluster_label_smoothed` (post-smoothing)
   - `context_confidence_score` ([0, 1])
   - `reconstruction_error` (anomaly proxy)
   - `is_anomaly` (binary flag)

3. **`cluster_profiles.csv`**: Per-cluster feature interpretations
   - `cluster_label, windows, top_positive_features, top_negative_features, mean_abs_z`

4. **`context_discovery_metrics.json`**: Run metadata
   ```json
   {
     "best_embedding": "umap",
     "best_model": "kmeans",
     "best_parameter": "k=6",
     "n_clusters": 6,
     "silhouette": 0.58,
     "davies_bouldin": 0.72,
     "calinski_harabasz": 1250.4,
     "stability_score": 0.85,
     "transition_score": 0.72,
     "anomaly_ratio": 0.053
   }
   ```

5. **Saved Artifacts** (`models/context_discovery/`):
   - `scaler.joblib`
   - `pca.joblib`
   - `umap_reducer.joblib`
   - `best_cluster_model.joblib`
   - `autoencoder.keras`

---

## **NOTEBOOK 05: Driver Behavior Modeling**

### Input
- `data/features/final_features_with_context.csv`
  - trip_id, window_id, cluster_label, cluster_label_smoothed, context_confidence_score, reconstruction_error, is_anomaly + engineered features

### Processing

#### Phase 1: Sequence Construction

**Trip-Stratified Sampling**
- Target: 80K windows
- Preserve behavior diversity via within-trip cluster stratification
- 70% train trips, 30% test trips (no leakage)

**Sequence Building:**
$$S_i = \{x_{t}, x_{t+1}, \ldots, x_{t+L-1}\}, \quad L = 20 \text{ steps}$$
- Sequence length: 20 windows (10 seconds @ effective 2Hz rate)
- Per trip: slide with stride 1 → all valid sequences

**Markov Feature Engineering:**

For context sequence $c_1, c_2, \ldots, c_L$:

**Transition matrix:**
$$T_{ij} = P(c_{t+1}=j | c_t=i)$$

**Transition entropy:**
$$H_t = -\sum_i P(i) \sum_j T_{ij} \log(T_{ij} + \epsilon)$$

**Temporal consistency:**
$$1 - \frac{\text{# context changes}}{L-1}$$

**Dwell time:**
$$\text{mean}(\text{lengths of constant-context segments})$$

**Sequence entropy:**
$$H_s = -\sum_k p_k \log(p_k + \epsilon), \quad p_k = P(c=k)$$

**Sequence log-likelihood:**
$$\sum_{t=1}^{L-1} \log(T_{c_t, c_{t+1}} + \epsilon)$$

**Multi-Input Tuple (per sequence):**
- `X_feat[t:t+L]`: (L, n_features) engineered features
- `X_ctx[t:t+L]`: (L,) context cluster IDs
- `X_sig[t:t+L]`: (L, 3) signals = [confidence, recon_error, is_anomaly]

**Targets** (predict one step ahead):
- `y_ctx[t+L]`: Next context (categorical)
- `y_feat[t+L]`: Next feature vector (regression)

#### Phase 2: Neural Sequence Encoder - BiLSTM-Attention

**Architecture:**
```
INPUTS:
  feat_input: (seq_len, n_features)
  ctx_input: (seq_len,) → int32
  sig_input: (seq_len, 3)

EMBEDDING LAYER:
  ctx_emb = Embedding(n_contexts+1, 16)(ctx_input)      → (seq_len, 16)
  feat_enc = Dense(96, relu)(feat_input)                 → (seq_len, 96)
  sig_enc = Dense(16, relu)(sig_input)                   → (seq_len, 16)

FUSION:
  x = Concat([feat_enc, ctx_emb, sig_enc])              → (seq_len, 128)

POSITIONAL ENCODING:
  pe[t, d] = sin(t / 10000^(2d/D))  for d even
  pe[t, d] = cos(t / 10000^(2(d-1)/D))  for d odd
  x = x + pe

SEQUENCE ENCODER (BiLSTM):
  x = Bidirectional(LSTM(64, return_seq=True))(x)      → (seq_len, 128)
  Dropout(0.2), recurrent_dropout(0.1)

ATTENTION POOLING:
  attn_logits = Dense(1, tanh)(x)                       → (seq_len, 1)
  attn_weights = Softmax(axis=1)(attn_logits)          → (seq_len, 1)
  context_vec = x * attn_weights                        → (seq_len, 128)
  embedding = ReduceSum(context_vec, axis=0)           → (128,)

PROJECTION HEAD (L2 normalized):
  proj = Dense(128, relu)(embedding)
  proj = LayerNorm(proj)
  proj_final = Dense(64)(proj)                          → (64,)

DECODER (reconstruction):
  rep = RepeatVector(seq_len)(embedding)               → (seq_len, 128)
  dec = Bidirectional(LSTM(64, return_seq=True))(rep)  → (seq_len, 128)
  recon = Dense(n_features)(dec)                       → (seq_len, n_features)

PREDICTION HEADS:
  h = Dense(128, relu)(embedding)
  h = BatchNorm(h)
  h = Dropout(0.2)(h)
  
  next_context = Dense(n_contexts, softmax)(h)         → (n_contexts,)
  next_features = Dense(n_features)(h)                 → (n_features,)

OUTPUTS:
  1. recon_seq: Reconstructed feature sequence
  2. next_context: Next-step context probability
  3. next_features: Next-step feature prediction
```

**Training Parameters:**
- Optimizer: Adam(lr=1e-3, clipnorm=1.0)
- Loss weights: {recon_seq: 1.0, next_context: 0.8, next_features: 0.8}
- Batch size: 256
- Epochs: 12 (early stopping: patience=3)
- Learning rate schedule: ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5)

#### Phase 3: Risk Scoring Framework

**Component 1: Anomaly Score**
$$\text{anomaly} = \text{normalize}(\text{reconstruction_error})$$

**Component 2: Instability Score**
$$\text{instability} = \frac{\text{# context changes in sequence}}{L}$$

**Component 3: Behavioral Score**
$$\text{behavior} = \text{context_confidence_score}$$
(How "prototypical" the sequence is)

**Component 4: Uncertainty Score**
$$\text{uncertainty} = -\sum_{k} p_k \log(p_k + \epsilon)$$
(Entropy of next_context_softmax, normalized to [0, 1])

**Component 5: Temporal Score**
$$\text{temporal} = 1 - \frac{\sum_{t=1}^{L-1} |\Delta \text{speed}[t]|}{\text{max_observed_change}}$$
(Inverse of erratic dynamics)

**Component 6: Transition Stability**
$$\text{transition_stability} = \sum_{i,j} T_{ij}^2$$
(Herfindahl index of transition matrix)

**Final Risk Score (Trip-level aggregation):**
Per window:
$$\text{risk}[t] = \text{anomaly} + \text{instability} + (1 - \text{behavior}) + \text{uncertainty} + (1 - \text{temporal})$$

Per trip:
$$\text{risk}_{\text{trip}} = \text{mean}(\text{risk}[\text{trip}])$$

Per driver (ensemble of trips):
$$\text{risk}_{\text{driver}} = \text{mean}(\text{risk}_{\text{trip}}) + \frac{\text{std}(\text{risk}_{\text{trip}})}{10}$$
(Adds inter-trip variability penalty)

#### Phase 4: Contrastive Learning (Optional)

Temporal pairs (adjacent sequences in same trip):
$$L_{\text{contrastive}} = \frac{1}{B} \sum_{i=1}^B -\log \frac{\exp(\text{sim}(a_i, p_i)/\tau)}{\sum_{j=1}^{2B} \exp(\text{sim}(a_i, j)/\tau)}$$
- Temperature: $\tau = 0.1$
- In-batch negatives
- Fixed tail layer + contrastive head trained 3 epochs

### Output Files

1. **`final_driver_behavior.csv`**: Sequence-level scores
   - `trip_id, window_id, risk_score, anomaly_score, behavior_score, uncertainty_score, temporal_score, sequence_instability, transition_stability`
   - `embedding_*`: 128-dim embeddings
   - `embedding_projection_*`: 64-dim contrastive projections

2. **`driver_profile.csv`**: Trip-level profiles
   - `trip_id, mean_risk, mean_anomaly, mean_instability, mean_uncertainty, mean_temporal, mean_behavior, windows`

3. **`embedding_metrics.csv`**: Quality metrics (silhouette, db_index, ch_score)

4. **`feature_importance.csv`**: Feature contribution scores

5. **`models/driver_behavior_v4/`** artifacts:
   - `scaler.joblib`
   - Best trained model weights

---

## **NOTEBOOK 06: Visualization and Reporting**

### Input
- `data/features/final_driver_behavior.csv`
- `data/features/driver_profile.csv`
- `data/features/embedding_metrics.csv`

### Processing

#### Section 1: Data Integration & Validation
- Merge sequence-level behavior with trip-level profiles
- Dynamic embedding column discovery
- Schema validation (all scores in [0, 1])
- Minimal NaN imputation (median strategy)

#### Section 2: Advanced Risk Analysis

**Visualizations:**

1. **Global Risk Distribution**
   - Histogram + KDE density plot
   - Identify risk sub-populations

2. **Risk by Context**
   - Stratified KDE by behavior_cluster
   - Shows context-specific risk profiles

3. **Risk by Driver Cluster** (violin plots)
   - Within-driver behavior variance

4. **Risk-Anomaly Interaction** (2D scatter + heatmap)
   - Confidence bands (lower/upper bounds)
   - Confidence zones: confident_safe, confident_risky, uncertain

**Statistics:**
- Top 5% risk threshold (P95)
- Bottom 5% safe threshold (P5)
- Driver leaderboard (mean_risk per trip)

#### Section 3: Temporal Intelligence

**1. Temporal Evolution Plots** (per selected trips)
- Multi-line chart: risk_score, temporal_score, sequence_instability, transition_stability
- Reveals behavior phase transitions

**2. Behavior Phase Heatmap**
- Rows: individual trips (sorted by instability)
- Columns: time index
- Values: behavior_phase_id (color-coded)

**3. Phase Transition Frequency Distribution**
- Histogram of mean_phase_freq (trip-level)
- Identifies drivers with frequent behavior changes

#### Section 4: Embedding Visualization

**1. 2D Projections** (with sampling for efficiency)
- **UMAP**:
  - `n_neighbors=30, min_dist=0.08`
  - Colored by risk_score, behavior_cluster, anomaly_score
  
- **t-SNE** (secondary):
  - `perplexity=30, learning_rate='auto'`
  - Similar stratifications

**2. Density Diagnostics**
- Local density heatmap (k-NN based)
- Reveals embedding space coverage

#### Section 5: Driver Leaderboard

**Output: `driver_leaderboard.csv`**

Columns per trip:
- `trip_id`
- `mean_risk` (primary ranking)
- `mean_anomaly`
- `mean_instability`
- `mean_uncertainty`
- `mean_temporal`
- `mean_behavior`
- `mean_transition_stability`
- `windows`
- Risk percentile
- Category: "High Risk" (P90+), "Medium", "Safe" (P10-)

### Output Reports

1. **`final_driver_analysis.csv`**: Merged sequence-trip insights

2. **`driver_leaderboard.csv`**: Sortable driver rankings

3. **`insights_summary.json`**: Key findings
   ```json
   {
     "total_trips": 50,
     "total_windows": 12000,
     "mean_global_risk": 0.52,
     "std_global_risk": 0.18,
     "high_risk_drivers": 5,
     "safe_drivers": 8,
     "top_risk_factors": ["sequence_instability", "anomaly_score", "uncertainty_score"],
     "confidence_zone_distribution": {
       "confident_safe": 0.35,
       "uncertain": 0.45,
       "confident_risky": 0.20
     }
   }
   ```

4. **Visualizations** (PNG/interactive HTML):
   - Risk distributions
   - Temporal evolution (sample trips)
   - Embedding projections
   - Transition matrices
   - Feature importance heatmaps
   - Leaderboard charts

---

## **Quick Reference: Notebook Pipeline**

| # | Notebook | Input | Processing | Key Methods | Output |
|---|----------|-------|-----------|-------------|--------|
| 1 | Data Cleaning | Raw OBD CSVs | Normalization, gap detection, interpolation, outlier clipping | Adaptive threshold (3×median Δt), per-trip scaling | clean_obd_data.csv |
| 2 | EDA | clean_obd_data.csv | Distributions, correlations, time-series plots | 60-bin histograms, Pearson correlation, scatter plots | Visualizations only |
| 3 | Feature Engineering | clean_obd_data.csv | Windowing (5s @ 1Hz, stride 2s), 110+ statistical/dynamic features | Per-window aggregations, z-score interactions, FFT entropy | feature_dataset.csv |
| 4 | Context Discovery | feature_dataset.csv | 4 embeddings (Raw, PCA, UMAP, AE) × 3 algorithms (KMeans, GMM, DBSCAN) | Silhouette, Davies-Bouldin, Calinski-Harabasz scoring | context_labels.csv, profiles, stability metrics |
| 5 | Modeling | final_features_with_context.csv | Sequence construction (L=20), BiLSTM-Attention encoding, multi-task learning | Markov features, attention pooling, multi-component risk scoring | embeddings, risk_scores, driver_profile.csv |
| 6 | Visualization | final_driver_behavior.csv | Risk stratification, 2D embeddings (UMAP/t-SNE), leaderboards | KDE distributions, violin plots, temporal heatmaps | leaderboard.csv, insights_summary.json, dashboards |

---

## **Key Findings & Insights**

- **Risk compositionality**: Risk co-varies with uncertainty and anomaly evidence, not a single factor
- **Embedding quality**: Moderate separability (silhouette ~0.58) with meaningful trip organization
- **Distribution asymmetry**: 89.66% windows fall in uncertain zones; only 1.49% confident-risky
- **Top risk features**: Throttle-related features (throttle_mean_scaled, throttle_std) rank highest among risk associations
- **Context stability**: Transition scores ~0.72 indicate meaningful cluster persistence
- **Anomaly ratio**: ~5.3% of windows flagged as anomalies based on reconstruction error

---

**Document Created:** May 2026
**Status:** Comprehensive technical reference for all 6 notebooks
**Ready for:** PPT preparation, project presentations, technical documentation
