import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import torch
import joblib
from model import OrbitLSTM
import os

def plot_3d_orbit(df_nominal, df_anomaly=None):
    fig = go.Figure()

    # Earth sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    r_earth = 6378.137
    x = r_earth * np.outer(np.cos(u), np.sin(v))
    y = r_earth * np.outer(np.sin(u), np.sin(v))
    z = r_earth * np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Blues', opacity=0.3, showscale=False, name='Earth'))

    # Nominal Orbit
    fig.add_trace(go.Scatter3d(x=df_nominal['x'], y=df_nominal['y'], z=df_nominal['z'],
                               mode='lines', line=dict(color='cyan', width=4), name='Nominal Orbit'))

    if df_anomaly is not None:
        fig.add_trace(go.Scatter3d(x=df_anomaly['x'], y=df_anomaly['y'], z=df_anomaly['z'],
                                   mode='lines', line=dict(color='red', width=4, dash='dot'), name='Anomaly Orbit'))

    fig.update_layout(
        title='OrbitMind: 3D Orbital Visualization',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            bgcolor='rgb(10, 10, 20)'
        ),
        paper_bgcolor='rgb(10, 10, 20)',
        font=dict(color='white'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    os.makedirs('docs', exist_ok=True)
    fig.write_html('docs/orbit_3d.html')
    print("3D Visualization saved to docs/orbit_3d.html")

def plot_prediction_error(y_true, y_pred, feature_name='x'):
    df = pd.DataFrame({
        'True': y_true,
        'Predicted': y_pred,
        'Error': np.abs(y_true - y_pred)
    })
    
    fig = px.line(df, title=f'Prediction Accuracy for {feature_name}')
    fig.update_layout(template='plotly_dark')
    fig.write_html(f'docs/prediction_{feature_name}.html')
    print(f"Prediction chart saved to docs/prediction_{feature_name}.html")

if __name__ == "__main__":
    # Load data
    df_nom = pd.read_csv('data/orbit_nominal.csv')
    df_ano = pd.read_csv('data/orbit_anomaly.csv')
    plot_3d_orbit(df_nom, df_ano)
