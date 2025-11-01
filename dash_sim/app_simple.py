"""Simple Race Replay Dashboard with Ribbon Offsets."""

import logging
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np

import config
from data_loader_simple import load_ribbons, load_telemetry_simple, prepare_trajectories

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = config.APP_TITLE

logger.info("=" * 60)
logger.info("RACE REPLAY DASHBOARD - SIMPLE RIBBON VERSION")
logger.info("=" * 60)

# Load ribbons
ribbons_data = load_ribbons(config.RIBBONS_FILE)
logger.info(f"Loaded {len(ribbons_data['ribbons'])} ribbons")

# Load telemetry
df = load_telemetry_simple(config.PARQUET_PATH, config.DEFAULT_CARS)

# Prepare trajectories
store_data = prepare_trajectories(df, ribbons_data, config.DEFAULT_RIBBON_BY_CAR)
logger.info(f"Prepared {len(store_data['car_ids'])} car trajectories")

# Get bounds for visualization
center_ribbon = next(r for r in ribbons_data['ribbons'] if r['name'] == 'center')
points = np.array(center_ribbon['xy'])
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()
logger.info(f"Track bounds: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")


def create_figure():
    """Create Plotly figure with ribbons and car markers."""
    fig = go.Figure()

    # Add all ribbons (thin, low opacity)
    for ribbon in ribbons_data['ribbons']:
        xy = np.array(ribbon['xy'])

        if ribbon['name'] == 'center':
            # Centerline: white, thicker, higher opacity
            fig.add_trace(go.Scattergl(
                x=xy[:, 0],
                y=xy[:, 1],
                mode='lines',
                line=dict(width=2, color='white'),
                opacity=0.6,
                name='Centerline',
                showlegend=False,
                hoverinfo='skip'
            ))
        else:
            # Offset ribbons: gray, thin, low opacity
            fig.add_trace(go.Scattergl(
                x=xy[:, 0],
                y=xy[:, 1],
                mode='lines',
                line=dict(width=1, color='gray'),
                opacity=0.2,
                name=ribbon['name'],
                showlegend=False,
                hoverinfo='skip'
            ))

    # Add car markers (initially at frame 0)
    for car_id in store_data['car_ids']:
        traj = store_data['trajectories'][car_id]

        # Get initial position
        x0 = traj['x'][0] if not np.isnan(traj['x'][0]) else 0
        y0 = traj['y'][0] if not np.isnan(traj['y'][0]) else 0

        fig.add_trace(go.Scattergl(
            x=[x0],
            y=[y0],
            mode='markers',
            marker=dict(
                size=config.CAR_MARKER_SIZE,
                color=traj['color'],
                opacity=config.CAR_MARKER_OPACITY,
                line=dict(width=1, color='white')
            ),
            name=f"Car {traj['car_no']}",
            hovertemplate=(
                f"<b>Car {traj['car_no']}</b><br>" +
                f"Ribbon: {traj['ribbon']}<br>" +
                "Position: (%{x:.0f}m, %{y:.0f}m)<br>" +
                "<extra></extra>"
            )
        ))

    # Set layout with equal aspect ratio
    padding = 50  # meters
    fig.update_xaxes(
        range=[x_min - padding, x_max + padding],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        range=[y_min - padding, y_max + padding],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#2a2a2a',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(
            text=config.APP_TITLE,
            x=0.5,
            xanchor='center'
        ),
        height=800,
        uirevision='constant',  # Preserve zoom/pan state
    )

    return fig


# App layout
app.layout = html.Div([
    # Hidden stores for data
    dcc.Store(id='store-trajectories', data=store_data),
    dcc.Store(id='store-state', data={'frame': 0, 'playing': False, 'speed': 1}),

    # Main content
    html.Div([
        # Track visualization
        dcc.Graph(
            id='track-graph',
            figure=create_figure(),
            config={'displayModeBar': True, 'scrollZoom': True}
        ),

        # Controls
        html.Div([
            html.Button('▶ Play', id='btn-play', n_clicks=0),
            html.Button('⏸ Pause', id='btn-pause', n_clicks=0),

            html.Label('Speed:', style={'margin-left': '20px'}),
            dcc.Dropdown(
                id='speed-dropdown',
                options=[
                    {'label': '0.5x', 'value': 0.5},
                    {'label': '1x', 'value': 1},
                    {'label': '2x', 'value': 2},
                    {'label': '4x', 'value': 4},
                ],
                value=1,
                clearable=False,
                style={'width': '100px', 'display': 'inline-block', 'margin-left': '10px'}
            ),

            html.Div(id='frame-info', style={'margin-left': '20px', 'display': 'inline-block'}),
        ], style={'padding': '10px', 'background': '#2a2a2a', 'display': 'flex', 'align-items': 'center'}),

        # Timeline slider
        dcc.Slider(
            id='frame-slider',
            min=0,
            max=store_data['frame_count'] - 1,
            value=0,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": False},
        ),

        # Animation ticker
        dcc.Interval(id='ticker', interval=config.TICK_INTERVAL_MS, disabled=True),

    ], style={'max-width': '1400px', 'margin': '0 auto'}),

], style={'background': '#2a2a2a', 'min-height': '100vh'})


# Callbacks
@app.callback(
    Output('ticker', 'disabled'),
    Output('store-state', 'data'),
    Input('btn-play', 'n_clicks'),
    Input('btn-pause', 'n_clicks'),
    Input('frame-slider', 'value'),
    Input('speed-dropdown', 'value'),
    State('store-state', 'data'),
    prevent_initial_call=True
)
def control_playback(play_clicks, pause_clicks, slider_value, speed, state):
    """Handle play/pause/seek/speed controls."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update

    trigger = ctx.triggered[0]['prop_id']

    if 'btn-play' in trigger:
        state['playing'] = True
    elif 'btn-pause' in trigger:
        state['playing'] = False
    elif 'frame-slider' in trigger:
        state['frame'] = slider_value
        state['playing'] = False  # Pause on manual seek
    elif 'speed-dropdown' in trigger:
        state['speed'] = speed

    return not state['playing'], state


@app.callback(
    Output('frame-slider', 'value'),
    Output('store-state', 'data', allow_duplicate=True),
    Input('ticker', 'n_intervals'),
    State('store-state', 'data'),
    State('store-trajectories', 'data'),
    prevent_initial_call=True
)
def animate_frame(n_intervals, state, traj_data):
    """Advance frame when playing."""
    if not state.get('playing', False):
        raise dash.exceptions.PreventUpdate

    # Advance frame
    new_frame = state['frame'] + int(state.get('speed', 1))
    if new_frame >= traj_data['frame_count']:
        new_frame = 0  # Loop back to start

    # Update state
    state['frame'] = new_frame

    return new_frame, state


@app.callback(
    Output('track-graph', 'figure'),
    Input('frame-slider', 'value'),
    State('store-trajectories', 'data'),
    State('track-graph', 'figure'),
    prevent_initial_call=True
)
def update_car_positions(frame_idx, traj_data, current_fig):
    """Update car marker positions when frame changes."""
    if current_fig is None:
        return dash.no_update

    # Ribbons are first N traces, cars are after that
    n_ribbons = len(ribbons_data['ribbons'])

    # Update each car trace
    for idx, car_id in enumerate(traj_data['car_ids']):
        traj = traj_data['trajectories'][car_id]
        x = traj['x'][frame_idx]
        y = traj['y'][frame_idx]

        # Only update if position is valid (check for None or NaN)
        if x is not None and y is not None:
            # Convert to float in case they're strings
            try:
                x_val = float(x)
                y_val = float(y)

                # Check if not NaN
                if not (np.isnan(x_val) or np.isnan(y_val)):
                    trace_idx = n_ribbons + idx
                    current_fig['data'][trace_idx]['x'] = [x_val]
                    current_fig['data'][trace_idx]['y'] = [y_val]
            except (ValueError, TypeError):
                # Skip invalid values
                pass

    return current_fig


@app.callback(
    Output('frame-info', 'children'),
    Input('store-state', 'data'),
    State('store-trajectories', 'data'),
)
def update_frame_info(state, traj_data):
    """Update frame counter display."""
    frame = state.get('frame', 0)
    total = traj_data['frame_count']
    time_sec = frame / config.TARGET_FPS

    return f"Frame: {frame:,} / {total:,} | Time: {int(time_sec // 60)}:{int(time_sec % 60):02d}"


if __name__ == '__main__':
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Starting dashboard on http://{config.APP_HOST}:{config.APP_PORT}")
    logger.info("=" * 60)

    app.run(
        host=config.APP_HOST,
        port=config.APP_PORT,
        debug=config.DEBUG
    )
