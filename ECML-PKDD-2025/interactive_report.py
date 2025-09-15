import dash
from dash import dcc, html, Input, Output
import json
import os
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from itertools import combinations


def _get_results(exp):
    report_path = os.path.join(exp, "results")
    results = []
    for file in os.listdir(report_path):
        if os.path.isfile(os.path.join(report_path, file)):
            with open(os.path.join(report_path, file), "r") as f:
                results.append(json.load(f))
    return results


def interactive_report(exp, params, metric='loss', log_loss=False, out_path=None):
    results = _get_results(exp)

    data = {param: np.zeros(len(results)) for param in params}
    data[metric] = np.zeros(len(results))

    for i in range(len(results)):
        for param in params:
            data[param][i] = results[i]['current_params'][param]
        data[metric][i] = results[i]['returned_dict'][metric]

    df = pd.DataFrame(data)

    param_pairs = list(combinations(params, 2))

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.Div([
            html.Label(f'Interval for {param} (log scale)'),
            dcc.RangeSlider(
                id=f'slider-{param}',
                min=np.log10(df[param].min())-0.1,
                max=np.log10(df[param].max())+0.1,
                step=0.01,
                value=[np.log10(df[param].min())-0.1, np.log10(df[param].max())+0.1],
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            dcc.Graph(
                id=f'scatter-{param}',
                style={
                    'height': '95%',
                    'width': '100%',
                    'margin': '0'
                }
            )
        ], style={
            'flex': '1',
            'height': '95vh',
            'padding': '0',
            'margin': '0',
            'boxSizing': 'border-box',
            'overflow': 'hidden'
        })

        for param in params
    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'width': '100vw',
        'height': '100vh',
        'margin': '0',
        'padding': '0',
        'overflow': 'hidden'
    })

    @app.callback(
        [Output(f'scatter-{param}', 'figure') for param in params],
        [Input(f'slider-{param}', 'value') for param in params]
    )
    def update_graphs(*intervals):
        limits = [(10 ** interval[0], 10 ** interval[1]) for interval in intervals]

        if out_path is not None:
            with open(os.path.join(out_path), 'w') as f:
                json.dump({param: [10 ** interval[0], 10 ** interval[1]] for param, interval in zip(params, intervals)}, f)

        insides = [
            (df[param] >= low) & (df[param] <= high)
            for param, (low, high) in zip(params, limits)
        ]

        inside_all = np.logical_and.reduce(insides)

        figs = []

        for param, (low, high) in zip(params, limits):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[param],
                y=df[metric],
                mode='markers',
                marker=dict(
                    color=df['loss'],  # Color mapped by rank
                    colorscale='Viridis',  # Color scale
                    size=[10 if inside_all[i] else 5 for i in range(len(df))],
                    opacity=[0.7 if inside_all[i] else 0.3 for i in range(len(df))]
                )
            ))
            fig.update_layout(
                #title=f'{metric} vs {param}',
                xaxis=dict(
                    title=param,
                    type='log',
                    tickformat=".0e",
                ),
                yaxis=dict(
                    title=metric,
                    type='log' if log_loss else 'linear',
                    tickformat=".0e" if log_loss else ".2f"
                ),
                shapes=[
                    dict(
                        type='rect',
                        x0=low,
                        x1=high,
                        y0=0,
                        y1=1,
                        yref='paper',
                        fillcolor='LightGray',
                        opacity=0.3,
                        layer='below',
                        line_width=0,
                    )
                ],
                margin=dict(l=0, r=0, t=0, b=100),
            )

            figs.append(fig)

        return figs

    app.run(debug=True)
