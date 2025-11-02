"""Race Replay Dashboard - Main Dash Application."""

import dash
from dash import dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go
from pathlib import Path
import logging
import argparse
import sys

import config
from data_loader_simple import load_ribbons, load_telemetry_simple, prepare_trajectories

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments for runtime configuration overrides."""
    parser = argparse.ArgumentParser(description="Race Replay Dashboard")
    parser.add_argument(
        '--cars',
        nargs='+',
        help='Chassis IDs to display (e.g., --cars 010 002)',
    )
    parser.add_argument(
        '--parquet',
        type=Path,
        help='Path to synchronized multi-car parquet file',
    )
    parser.add_argument(
        '--track',
        help='Track name (e.g., barber)',
    )
    return parser.parse_args()


# Parse CLI arguments and override config if provided
args = parse_args()

# Override config with CLI args if provided
PARQUET_PATH = args.parquet if args.parquet else config.PARQUET_PATH
CURRENT_TRACK = args.track if args.track else config.CURRENT_TRACK
DEFAULT_CARS = args.cars if args.cars else config.DEFAULT_CARS

# Initialize Dash app
app = dash.Dash(
    __name__,
    title=config.APP_TITLE,
    update_title=None,  # Disable "Updating..." in title
)

# Load data at startup
logger.info("=" * 60)
logger.info("RACE REPLAY DASHBOARD - INITIALIZING")
logger.info("=" * 60)
logger.info(f"Parquet: {PARQUET_PATH}")
logger.info(f"Ribbons: {config.RIBBONS_FILE}")
logger.info(f"Cars: {DEFAULT_CARS}")

try:
    # Load ribbons
    import numpy as np
    import json
    import pickle

    ribbons_data = load_ribbons(config.RIBBONS_FILE)
    logger.info(f"Loaded {len(ribbons_data['ribbons'])} ribbons")

    # Check for cached trajectories (track-aware)
    cache_file = Path(__file__).parent / "cache" / f"{CURRENT_TRACK}_trajectories_{'_'.join(DEFAULT_CARS)}.pkl"
    cache_file.parent.mkdir(exist_ok=True)

    if cache_file.exists():
        logger.info(f"Loading cached trajectories from {cache_file.name}...")
        with open(cache_file, 'rb') as f:
            store_data = pickle.load(f)
        logger.info(f"Loaded {len(store_data['car_ids'])} car trajectories from cache")
    else:
        logger.info("No cache found, computing trajectories...")
        # Load telemetry
        df = load_telemetry_simple(PARQUET_PATH, DEFAULT_CARS)

        # Prepare trajectories (disable GPS fallback for faster startup during development)
        store_data = prepare_trajectories(df, ribbons_data, config.DEFAULT_RIBBON_BY_CAR, use_gps_fallback=False)
        logger.info(f"Prepared {len(store_data['car_ids'])} car trajectories")

        # Save to cache
        logger.info(f"Saving trajectories to cache: {cache_file.name}")
        with open(cache_file, 'wb') as f:
            pickle.dump(store_data, f)

    # Get bounds for visualization
    center_ribbon = next(r for r in ribbons_data['ribbons'] if r['name'] == 'center')
    points = np.array(center_ribbon['xy'])
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    logger.info(f"Data loaded successfully: {store_data['frame_count']:,} frames")
    logger.info(f"Track bounds: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    raise


def create_initial_figure():
    """Create initial Plotly figure with track centerline, ribbons, and car traces."""
    fig = go.Figure()

    # Add ribbon traces (thin, low opacity)
    for ribbon in ribbons_data['ribbons']:
        if ribbon['name'] == 'center':
            # Centerline: white, thicker, higher opacity
            fig.add_trace(go.Scattergl(
                x=[pt[0] for pt in ribbon['xy']],
                y=[pt[1] for pt in ribbon['xy']],
                mode='lines',
                line=dict(width=2, color='white'),
                opacity=0.6,
                name='Track Centerline',
                showlegend=False,
                hoverinfo='skip'
            ))
        else:
            # Ribbons: gray, thin, low opacity
            fig.add_trace(go.Scattergl(
                x=[pt[0] for pt in ribbon['xy']],
                y=[pt[1] for pt in ribbon['xy']],
                mode='lines',
                line=dict(width=1, color='gray'),
                opacity=0.2,
                name=ribbon['name'],
                showlegend=False,
                hoverinfo='skip'
            ))

    # Add car marker traces (one scatter trace per car)
    import math
    def is_valid_xy(x, y):
        return (x is not None and y is not None and
                not (isinstance(x, float) and math.isnan(x)) and
                not (isinstance(y, float) and math.isnan(y)))

    for car_id in store_data['car_ids']:
        traj = store_data['trajectories'][car_id]

        # Find first valid position (not None and not NaN)
        x_arr = traj['x']
        y_arr = traj['y']
        first_valid_idx = next((i for i, (x, y) in enumerate(zip(x_arr, y_arr))
                               if is_valid_xy(x, y)), 0)

        vx = x_arr[first_valid_idx] if first_valid_idx < len(x_arr) and not (isinstance(x_arr[first_valid_idx], float) and math.isnan(x_arr[first_valid_idx])) else None
        vy = y_arr[first_valid_idx] if first_valid_idx < len(y_arr) and not (isinstance(y_arr[first_valid_idx], float) and math.isnan(y_arr[first_valid_idx])) else None

        initial_x = [vx] if vx is not None else []
        initial_y = [vy] if vy is not None else []

        fig.add_trace(go.Scattergl(
            x=initial_x,
            y=initial_y,
            mode='markers',
            name=f"Car {traj['car_no']}",
            marker=dict(
                size=config.CAR_MARKER_SIZE,
                color=traj['color'],
                opacity=config.CAR_MARKER_OPACITY,
                line=dict(width=2, color='white')
            ),
            hovertemplate=(
                f"<b>Car {traj['car_no']}</b><br>" +
                f"Ribbon: {traj.get('ribbon', 'N/A')}<br>" +
                "Position: (%{x:.0f}m, %{y:.0f}m)<br>" +
                "<extra></extra>"
            )
        ))

    # Add padding to bounds
    padding = 50  # meters
    x_range = [x_min - padding, x_max + padding]
    y_range = [y_min - padding, y_max + padding]

    # Configure layout
    fig.update_xaxes(
        range=x_range,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        range=y_range,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        scaleanchor="x",  # Maintain equal aspect ratio
        scaleratio=1,
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='black',
        plot_bgcolor='black',
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='white',
            borderwidth=1,
            font=dict(color='white', size=12)
        ),
        height=800,
    )

    return fig


# Create app layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1(config.APP_TITLE, style={'margin': '10px', 'color': 'white'}),
        html.Div(id='status-display', style={'margin': '10px', 'color': 'white', 'fontSize': '14px'}),
    ], style={'backgroundColor': '#1a1a1a', 'padding': '10px'}),

    # Main track visualization
    dcc.Graph(
        id='track-graph',
        figure=create_initial_figure(),
        config={'displayModeBar': False},
        style={'height': '800px'}
    ),

    # Control panel
    html.Div([
        # Play/Pause buttons
        html.Div([
            html.Button('▶ Play', id='btn-play', n_clicks=0,
                       style={'marginRight': '10px', 'padding': '10px 20px', 'fontSize': '16px'}),
            html.Button('⏸ Pause', id='btn-pause', n_clicks=0,
                       style={'marginRight': '20px', 'padding': '10px 20px', 'fontSize': '16px'}),
        ], style={'display': 'inline-block', 'marginRight': '20px'}),

        # Speed control
        html.Div([
            html.Label('Speed: ', style={'color': 'white', 'marginRight': '10px'}),
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
                style={'width': '100px', 'display': 'inline-block'}
            ),
        ], style={'display': 'inline-block', 'marginRight': '20px'}),

        # Frame info
        html.Div(id='frame-info', style={'display': 'inline-block', 'color': 'white', 'marginLeft': '20px'}),
    ], style={'backgroundColor': '#2a2a2a', 'padding': '15px'}),

    # Timeline slider
    html.Div([
        dcc.Slider(
            id='frame-slider',
            min=0,
            max=store_data['frame_count'] - 1,
            value=0,
            marks={
                0: '0:00',
                store_data['frame_count'] // 2: f"{store_data['frame_count'] // 2 // config.TARGET_FPS // 60}:{(store_data['frame_count'] // 2 // config.TARGET_FPS) % 60:02d}",
                store_data['frame_count'] - 1: f"{(store_data['frame_count'] - 1) // config.TARGET_FPS // 60}:{((store_data['frame_count'] - 1) // config.TARGET_FPS) % 60:02d}",
            },
            tooltip={"placement": "bottom", "always_visible": False},
        ),
    ], style={'backgroundColor': '#2a2a2a', 'padding': '15px 30px'}),

    # Hidden stores for state management
    dcc.Store(id='store-trajectories', data=store_data),
    dcc.Store(id='store-state', data={'frame': 0, 'playing': False, 'speed': 1}),

    # Animation ticker
    dcc.Interval(id='ticker', interval=config.TICK_INTERVAL_MS, n_intervals=0, disabled=True),

], style={'backgroundColor': '#1a1a1a', 'minHeight': '100vh'})


# Helper functions for callbacks
CAR_TRACES_START_INDEX = 11  # Car traces come after 11 ribbon traces

def _wrap_frame(n, frame_count):
    """Wrap frame index with bounds checking."""
    if not frame_count:
        return 0
    return n % frame_count


# Server-side callbacks

@app.callback(
    Output('store-state', 'data', allow_duplicate=True),
    Output('ticker', 'disabled'),
    Input('btn-play', 'n_clicks'),
    Input('btn-pause', 'n_clicks'),
    Input('speed-dropdown', 'value'),
    State('store-state', 'data'),
    prevent_initial_call=True
)
def control_playback(play_clicks, pause_clicks, speed, state):
    """Handle play/pause/speed controls (slider removed to avoid circular dependency)."""
    ctx = dash.callback_context

    if not ctx.triggered:
        return state, True

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'btn-play':
        logger.info("▶ PLAY")
        new_state = dict(state or {})
        new_state['playing'] = True
        new_state['start_frame'] = int(new_state.get('frame', 0))
        new_state['start_n'] = None  # Will be set on first tick
        logger.info(f"  Returning: state={new_state}, ticker.disabled=False")
        return new_state, False  # Enable ticker
    elif trigger_id == 'btn-pause':
        logger.info("⏸ PAUSE")
        new_state = dict(state or {})
        new_state['playing'] = False
        logger.info(f"  Returning: state={new_state}, ticker.disabled=True")
        return new_state, True  # Disable ticker
    elif trigger_id == 'speed-dropdown':
        logger.info(f"Speed: {speed}x")
        new_state = dict(state or {})
        new_state['speed'] = speed
        ticker_disabled = not new_state['playing']
        logger.info(f"  Returning: state={new_state}, ticker.disabled={ticker_disabled}")
        return new_state, ticker_disabled

    return state, not state['playing']


@app.callback(
    Output('track-graph', 'figure'),
    Input('store-state', 'data'),
    State('store-trajectories', 'data'),
    State('track-graph', 'figure'),
    prevent_initial_call=True
)
def update_graph_from_store(state, traj_data, current_fig):
    """Update car positions from store (driven by both animation and manual seek)."""
    frame_idx = state.get('frame', 0)
    if current_fig is None:
        return dash.no_update
    ribbon_count = len(ribbons_data['ribbons'])
    updated = 0
    import math
    for idx, car_id in enumerate(traj_data['car_ids']):
        traj = traj_data['trajectories'][car_id]
        x = traj['x'][frame_idx]
        y = traj['y'][frame_idx]
        if x is not None and y is not None and not (isinstance(x, float) and math.isnan(x)) and not (isinstance(y, float) and math.isnan(y)):
            trace_idx = ribbon_count + idx
            current_fig['data'][trace_idx]['x'] = [x]
            current_fig['data'][trace_idx]['y'] = [y]
            updated += 1
    if frame_idx % 20 == 0:
        logger.info(f"positions updated: {updated} cars @ frame {frame_idx}")
    return current_fig


@app.callback(
    Output('frame-info', 'children'),
    Input('ticker', 'n_intervals'),
    State('store-state', 'data'),
    State('store-trajectories', 'data'),
)
def heartbeat(n, state, traj):
    """Heartbeat to show ticker is firing and frame is advancing."""
    if n is None:
        return "Waiting for ticker..."
    f = int(state.get('frame', 0))
    total = int(traj.get('frame_count', 0))
    time_sec = f / config.TARGET_FPS
    logger.info(f"HEARTBEAT FIRED: n={n}, frame={f}")
    return f"Ticker={n} | frame={f:,} / {total:,} | Time: {int(time_sec // 60)}:{int(time_sec % 60):02d}"


# Server-side animation callback
@app.callback(
    Output('store-state', 'data', allow_duplicate=True),
    Input('ticker', 'n_intervals'),
    State('store-state', 'data'),
    State('store-trajectories', 'data'),
    prevent_initial_call=True
)
def animate_frame(n_intervals, state, traj):
    """Advance frame when playing - stateless computation from ticker anchors."""
    logger.info(f"animate_frame CALLED: n={n_intervals}, playing={state.get('playing')}")

    if not state.get('playing', False):
        logger.info("  Not playing - PreventUpdate")
        raise dash.exceptions.PreventUpdate

    total = int(traj['frame_count']) or 1
    speed = int(round(state.get('speed', 1))) or 1

    # Anchor setup: lock to ticker value when Play pressed
    start_n = state.get('start_n')
    start_frame = int(state.get('start_frame', state.get('frame', 0)))
    if start_n is None:
        start_n = int(n_intervals)

    elapsed = int(n_intervals) - int(start_n)
    frame = (start_frame + elapsed * speed) % total

    logger.info(f"  Returning frame={frame}")

    # Return FRESH dict (no mutation)
    return {**state, 'frame': int(frame), 'start_n': int(start_n), 'start_frame': int(start_frame)}


# SLIDER CALLBACKS DISABLED - They create circular dependency that pauses playback
# sync_slider_from_store updates slider → user_seek_slider detects change → sets playing=False
# We'll fix this after getting core animation working

# @app.callback(
#     Output('frame-slider', 'value', allow_duplicate=True),
#     Input('store-state', 'data'),
#     prevent_initial_call=True
# )
# def sync_slider_from_store(state):
#     """Mirror store frame to slider (read-only display)."""
#     return state.get('frame', 0)


# @app.callback(
#     Output('store-state', 'data', allow_duplicate=True),
#     Input('frame-slider', 'value'),
#     State('store-state', 'data'),
#     prevent_initial_call=True
# )
# def user_seek_slider(slider_value, state):
#     """Handle user manual scrubbing - pause playback and seek to frame."""
#     state = dict(state, playing=False, frame=int(slider_value or 0))
#     return state


@app.callback(
    Output('status-display', 'children'),
    Input('store-trajectories', 'data'),
)
def display_status(traj_data):
    """Display loading status and data info."""
    car_count = len(traj_data['car_ids'])
    frame_count = traj_data['frame_count']
    duration_sec = frame_count / config.TARGET_FPS
    return f"Loaded {car_count} cars | {frame_count:,} frames | Duration: {int(duration_sec // 60)}:{int(duration_sec % 60):02d}"


if __name__ == '__main__':
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting dashboard on http://{config.APP_HOST}:{config.APP_PORT}")
    logger.info(f"{'='*60}\n")

    app.run(
        host=config.APP_HOST,
        port=config.APP_PORT,
        debug=config.DEBUG
    )
