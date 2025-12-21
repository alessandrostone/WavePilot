After analyzing the code, here are several fast optimization strategies:

  Current Bottlenecks:

  1. 300 trials - Takes significant time even with parallelization
  2. Mahalanobis distance loop - Computed per-point instead of vectorized
  3. Covariance recomputation - Computed for every trial
  4. Large search space - 5 kernels × continuous params

  Faster Implementations (from fastest to best quality):

  Option 1: Skip Optimization (Instant)

  Use proven defaults - thin_plate_spline performs well in most cases:

  best_rbf_params = {
      "kernel": "thin_plate_spline",
      "smoothing": 1e-6,
      "epsilon_scale": 1.0,
      "degree": 1
  }

  Speedup: ~300x (0 seconds vs minutes)

  Option 2: Grid Search Kernels Only (10-20x faster)

  Test only the 5 kernel types with fixed optimal params:

  kernels_to_test = ["gaussian", "thin_plate_spline", "cubic",
                     "inverse_quadratic", "linear"]
  fixed_params = {"smoothing": 1e-6, "epsilon_scale": 1.0}
  # Test each kernel, pick best

  Speedup: ~15-60x (5-20 trials vs 300)

  Option 3: Vectorized Distance Calculation (2-3x faster)

  Pre-compute covariance and vectorize Mahalanobis:

  # Pre-compute once (outside trial loop)
  cov_matrix = np.cov(original_data, rowvar=False)
  inv_cov = np.linalg.inv(cov_matrix + 1e-12 * np.eye(cov_matrix.shape[1]))

  # Vectorized Mahalanobis (inside trial)
  diff = original_data - interpolated_data
  distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
  mean_distance = np.mean(distances)

  Speedup: 2-3x for the validation step

  Option 4: Reduced Trial Count (3-6x faster)

  Reduce from 300 to 50-100 trials:

  --rbf-trials 50  # vs default 300

  Speedup: 3-6x (proportional reduction)

  Recommended Approach:

  Combine Options 3 + 4 for 5-10x speedup with minimal quality loss:

  1. Vectorize distance calculation
  2. Reduce to 50-100 trials
  3. Pre-compute covariance matrix

  Would you like me to implement any of these optimizations?

> explain why thin_plate_spline should be fine in most cases

⏺ Why Thin Plate Spline (TPS) is the Best Default Choice

  1. Mathematical Foundation

  TPS minimizes the bending energy of the interpolating surface:

  E = ∫∫ (∂²f/∂x² + 2∂²f/∂x∂y + ∂²f/∂y²)² dx dy

  This means TPS finds the smoothest possible interpolation that passes through all points - like a thin metal plate bent to
  touch specific points with minimal strain.

  2. Perfect for Audio Parameter Spaces

  Audio parameters are naturally smooth:
  - Moving a filter cutoff from 1000Hz → 1200Hz should smoothly change the sound
  - Jumping discontinuously would create audible artifacts
  - TPS guarantees smooth transitions between presets

  Example:
  Preset A: cutoff=1000Hz, resonance=0.3
  Preset B: cutoff=2000Hz, resonance=0.7

  TPS interpolation: Smooth curve through parameter space
  Linear interpolation: Straight line (can miss sweet spots)
  Gaussian: Might create "dead zones" between presets

  3. Comparison to Other Kernels

  | Kernel            | Behavior                   | Problem for Audio                    |
  | ----------------- | -------------------------- | ------------------------------------ |
  | thin_plate_spline | ✅ Smooth, global influence | None - ideal                         |
  | gaussian          | 🔸 Localized "bumps"        | Creates isolated parameter regions   |
  | cubic             | 🔸 Similar to TPS           | Less theoretically motivated         |
  | linear            | ❌ No curvature             | Too simplistic, misses relationships |
  | inverse_quadratic | ⚠️ Can be unstable          | Numerical issues with close points   |

  4. Works Well with Latent Spaces

  Your VAE creates a 2D or 3D latent space:
  - TPS excels in low dimensions (2D/3D)
  - Handles scattered, irregular point distributions
  - No "grid" required - works with any preset layout

  5. Proven Track Record

  TPS is the standard for:
  - Shape morphing (3D graphics)
  - Image warping (computer vision)
  - Scattered data interpolation (scientific computing)
  - Geographical data (terrain modeling)

  All these applications share the same requirement: smooth interpolation through scattered points.

  6. Practical Evidence

  From your codebase constants:
  RBF_MIN_DEGREE = {
      "thin_plate_spline": 1,  # Requires polynomial term
      # This adds a linear trend, improving extrapolation
  }

  The polynomial term (degree=1) adds a global linear trend, making TPS even better for preset interpolation by capturing
  overall parameter relationships.

  When TPS Might Not Be Ideal

  Only in rare cases:
  1. Sharp transitions wanted - e.g., switching between completely different sound classes (use gaussian)
  2. Very sparse data (< 10 presets) - might overfit (use linear)
  3. Extreme outliers - TPS tries to connect everything smoothly (consider inverse_quadratic)

  For audio preset interpolation with 20-200 presets in a smooth latent space, TPS is the gold standard.

> and what about when a preset has parameters booleans or values from a list?

⏺ Excellent question! This is a critical real-world problem. Most plugins have mixed parameter types:

  The Problem

  1. Boolean Parameters (On/Off)

  Preset A: bypass=0.0 (OFF)
  Preset B: bypass=1.0 (ON)

  RBF interpolation: bypass=0.5  ❌ INVALID!
  # Plugin sees "half-bypassed" - undefined behavior

  2. Discrete List Parameters (Enums)

  Preset A: waveform=0.0    (Sine)
  Preset B: waveform=0.666  (Saw)
  Preset C: waveform=1.0    (Triangle)

  RBF interpolation: waveform=0.333  ❌
  # Plugin might interpret as "Square" but context wanted Sine→Saw

  3. Integer Steps

  Preset A: voices=4 (normalized: 0.25)
  Preset B: voices=12 (normalized: 0.75)

  RBF interpolation: 0.5 → voices=8  ✅ (valid but may sound awkward)

  Current Implementation Issue

  Your pipeline normalizes ALL parameters to [0,1] and interpolates continuously. This works perfectly for:
  - ✅ Filter cutoff
  - ✅ Resonance
  - ✅ Volume
  - ✅ Pan

  But fails for:
  - ❌ Boolean switches
  - ❌ Waveform selectors
  - ❌ Filter type (LP/HP/BP/Notch)
  - ❌ Oscillator count

  Solutions (Ranked by Practicality)

  Option 1: Post-Processing Quantization ⭐ (Recommended)

  What it does: Let RBF interpolate freely, then snap discrete parameters to valid values.

  def quantize_discrete_params(interpolated_values, param_types):
      result = interpolated_values.copy()

      for i, param_type in enumerate(param_types):
          if param_type == "boolean":
              # Round to 0 or 1
              result[i] = round(result[i])

          elif param_type == "discrete":
              # Snap to nearest valid value
              valid_values = param_types[i]["values"]  # e.g., [0.0, 0.33, 0.67, 1.0]
              result[i] = min(valid_values, key=lambda x: abs(x - result[i]))

      return result

  Pros:
  - Simple to implement
  - Works with existing RBF code
  - Fast (just post-processing)

  Cons:
  - Discontinuities when crossing thresholds
  - Example: Moving through latent space, bypass suddenly flips ON

  When to use: Default choice - works well enough for most cases

  ---
  Option 2: Exclude Discrete Parameters from RBF ⭐⭐

  What it does: Only interpolate continuous parameters, use nearest-neighbor for discrete ones.

  # During training: Mask discrete parameters
  continuous_mask = [i for i, p in enumerate(params) if p.type == "continuous"]
  discrete_mask = [i for i, p in enumerate(params) if p.type != "continuous"]

  # Train RBF only on continuous parameters
  rbf = RBFInterpolator(Z_latent, data[:, continuous_mask])

  # At runtime: Interpolate continuous, copy nearest discrete
  interpolated_continuous = rbf(new_latent_point)
  nearest_preset_idx = find_nearest(new_latent_point, preset_latent_points)
  discrete_values = presets[nearest_preset_idx][discrete_mask]

  final_params = merge(interpolated_continuous, discrete_values)

  Pros:
  - No invalid states
  - Discrete params change intentionally (when closest preset changes)

  Cons:
  - More complex
  - Loses smoothness at preset boundaries
  - Requires knowing parameter types upfront

  When to use: When you have many discrete parameters and want clean behavior

  ---
  Option 3: Weighted Nearest-Neighbor for Discrete ⭐⭐⭐ (Best Quality)

  What it does: Use distance-weighted voting among nearby presets for discrete parameters.

  # Find 3 nearest presets in latent space
  nearest_presets, distances = find_k_nearest(new_latent_point, presets, k=3)

  # For discrete params: weighted voting
  for discrete_param in discrete_params:
      values = [preset[discrete_param] for preset in nearest_presets]
      weights = 1.0 / (distances + 1e-6)  # Closer = higher weight

      # Most common value, weighted by distance
      param_value = weighted_mode(values, weights)

  Pros:
  - Smooth continuous interpolation
  - Intelligent discrete parameter selection
  - Mimics "musical intent" - if 2 nearby presets use "Saw", probably want Saw

  Cons:
  - Complex implementation
  - Requires tracking preset metadata

  When to use: Production-quality preset morphing systems

  ---
  Option 4: TPS with Discontinuity Handling (Research-Level)

  Modify TPS kernel to handle discontinuous dimensions:

  # Split parameter space: continuous × discrete
  # Use TPS for continuous, categorical distance for discrete

  Very complex - not recommended unless you're writing a paper.

  ---
  Practical Recommendation for WavePilotHost:

  Phase 1: Post-Processing Quantization (Now)

  Add a simple quantization step in your C++ RBF code:

  // In HostAudioProcessorImpl.cpp after RBF interpolation
  for (int i = 0; i < numParams; ++i) {
      auto* param = parameters[i];

      // Check if parameter is discrete
      if (param->isDiscrete()) {
          float value = interpolatedValues[i];

          // Snap to nearest valid step
          int numSteps = param->getNumSteps();
          value = std::round(value * numSteps) / numSteps;
          interpolatedValues[i] = value;
      }
  }

  Phase 2: Mask Discrete Parameters (Future)

  Allow users to exclude certain parameters from interpolation:

  # In pipeline.py
  python pipeline.py -i presets.csv -o model.json \
      --discrete-params "bypass,waveform_type,filter_mode"

  Why This Matters

  Without handling discrete parameters:
  - ❌ Plugin might crash (invalid parameter combinations)
  - ❌ Sound artifacts (half-bypassed states)
  - ❌ Confusing UX (waveform flickering between types)

  With proper handling:
  - ✅ Always valid parameter states
  - ✅ Smooth continuous morphing
  - ✅ Intentional discrete changes
