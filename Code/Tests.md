Fragen, die durch Tests beantwortet werden sollen:
1. Sensitivity to non-stationary behavior of the spectral signatures of land-cover classes
  in the spatial domain of the scene, which is due to physical factors related to ground
  (e.g., different soil moisture or composition), vegetation, and atmospheric conditions
2. Sensitivity to Hughes Phenomenom (small ratio between No of training samples and No of spectral channels, originally not for ANNs)
3. Ist 1D+2D besser als nur 1D? (Für das Dataset)
4. Kann 1D-Modell verbessert werden durch bessere Parameter (Filter nrs / Kernel sizes)
5. Pattern redundancies durch hyperprior (?)
6. Vergleich der Latents mit LPIPS
7. Vergleich von Modell mit Frozen Spectral Autoencoder zu Nicht frozen (+ nur Inner Loss) -> Zeigt ob Decoder = Inverse of Encoder
8. Test 1D vs Simple interpolation
9. JPEG (2000) als Baseline
10. Stopping Criterion basierend auf wenig Veränderung im loss
