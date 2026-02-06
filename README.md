# OrbitMind 🚀
**Deep Learning for Satellite Orbit Prediction & Anomaly Detection**

OrbitMind is a research-oriented project that leverages Recurrent Neural Networks (LSTMs) to predict satellite trajectories and detect orbital anomalies (maneuvers, sensor noise, or decay).

## 🌌 Scientific Motivation
Space situational awareness (SSA) is critical as Earth's orbit becomes more crowded. Traditionally, orbits are propagated using numerical integrators (like SGP4 for TLEs). OrbitMind combines classical orbital mechanics with Deep Learning to:
- Predict future positions with high accuracy.
- Identify unauthorized maneuvers or sudden changes in orbital state.
- Handle noisy sensor data using reconstruction techniques.

## 🛠 Project Components
- **Data Propagator**: Uses the Two-Body Problem equations $\ddot{\vec{r}} = -\frac{\mu}{r^3} \vec{r}$ to generate realistic synthetic orbits.
- **LSTM Predictor**: A stacked LSTM network trained on time-series position/velocity vectors $(x, y, z, v_x, v_y, v_z)$.
- **Anomaly Detection**: Uses prediction error thresholds (MSE) to flag deviations from expected physical paths.
- **Physics-Informed Loss**: Optional constraint that penalizes models for violating Conservation of Specific Mechanical Energy $E = \frac{v^2}{2} - \frac{\mu}{r}$.

## 🚀 How it Works
1. **Generation**: We simulate a satellite in LEO (400km altitude).
2. **Preprocessing**: Data is normalized and windowed (Lookback: 20 timestamps).
3. **Training**: The model learns the periodicity and dynamics of the central force field.
4. **Inference**: We inject a "Maneuver" (sudden $\Delta V$) and the model flags it as an anomaly when the prediction error spikes.

## 📊 Visualizations
The project generates interactive 3D visualizations of the orbit around Earth, helping researchers visualize path deviations.

---
*Created for Space AI Research & Aerospace Portfolios.*
