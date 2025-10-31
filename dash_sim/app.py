"""Race Replay Dashboard - Main Dash Application."""

import dash
from dash import dcc, html, Input, Output, State, ClientsideFunction
import plotly.graph_objects as go
from pathlib import Path
import logging

import config
from data_loader import load_and_prepare_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

try:
    store_data, transformer = load_and_prepare_data(
        config.PARQUET_PATH,
        config.TRACK_IMAGE,
        config.DEFAULT_CARS
    )
    img_width, img_height = transformer.get_image_dimensions()
    logger.info(f"Data loaded successfully: {store_data['frame_count']:,} frames")
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    raise


def create_initial_figure():
    """Create initial Plotly figure with track background and car traces."""
    fig = go.Figure()

    # Add track background image
    fig.add_layout_image(
        dict(
            source=f"/{config.TRACK_IMAGE.name}",
            xref="x",
            yref="y",
            x=0,
            y=img_height,
            sizex=img_width,
            sizey=img_height,
            sizing="stretch",
            opacity=1.0,
            layer="below"
        )
    )

    # Add initial car traces (one scatter trace per car)
    for car_id in store_data['car_ids']:
        traj = store_data['trajectories'][car_id]

        # Find first valid position
        x_arr = traj['x']
        y_arr = traj['y']
        first_valid_idx = next((i for i, (x, y) in enumerate(zip(x_arr, y_arr))
                               if not (x is None or y is None)), 0)

        initial_x = [x_arr[first_valid_idx]] if first_valid_idx < len(x_arr) else [0]
        initial_y = [y_arr[first_valid_idx]] if first_valid_idx < len(y_arr) else [0]

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
                "Position: (%{x:.0f}, %{y:.0f})<br>" +
                "<extra></extra>"
            )
        ))

    # Configure layout
    fig.update_xaxes(
        range=[0, img_width],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        range=[0, img_height],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
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


# Server-side callbacks

@app.callback(
    Output('store-state', 'data'),
    Output('ticker', 'disabled'),
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
        return state, True

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'btn-play':
        state['playing'] = True
        return state, False  # Enable ticker
    elif trigger_id == 'btn-pause':
        state['playing'] = False
        return state, True  # Disable ticker
    elif trigger_id == 'frame-slider':
        state['frame'] = slider_value
        state['playing'] = False  # Pause when seeking
        return state, True
    elif trigger_id == 'speed-dropdown':
        state['speed'] = speed
        return state, not state['playing']

    return state, not state['playing']


@app.callback(
    Output('frame-info', 'children'),
    Input('store-state', 'data'),
    Input('store-trajectories', 'data'),
)
def update_frame_info(state, traj_data):
    """Update frame counter display."""
    frame = state['frame']
    total = traj_data['frame_count']
    time_sec = frame / config.TARGET_FPS
    return f"Frame: {frame:,} / {total:,} | Time: {int(time_sec // 60)}:{int(time_sec % 60):02d}"


# Client-side callback for smooth animation
app.clientside_callback(
    """
    function(n_intervals, state, trajectories) {
        if (!state.playing) {
            return window.dash_clientside.no_update;
        }

        // Advance frame
        let newFrame = state.frame + Math.round(state.speed);
        if (newFrame >= trajectories.frame_count) {
            newFrame = 0;  // Loop
        }

        // Update slider
        const sliderUpdate = newFrame;

        // Update figure traces (car positions)
        const figure = window.Plotly.data('track-graph');
        const update = {x: [], y: []};

        trajectories.car_ids.forEach((carId, idx) => {
            const traj = trajectories.trajectories[carId];
            const x = traj.x[newFrame];
            const y = traj.y[newFrame];

            // Only update if position is valid (not null/NaN)
            if (x !== null && y !== null && !isNaN(x) && !isNaN(y)) {
                update.x.push([x]);
                update.y.push([y]);
            } else {
                // Keep previous position if current is invalid
                update.x.push([figure[idx].x[0]]);
                update.y.push([figure[idx].y[0]]);
            }
        });

        // Update all traces
        Plotly.update('track-graph', update, {}, Array.from({length: trajectories.car_ids.length}, (_, i) => i));

        // Update state
        state.frame = newFrame;

        return [sliderUpdate, state];
    }
    """,
    Output('frame-slider', 'value'),
    Output('store-state', 'data', allow_duplicate=True),
    Input('ticker', 'n_intervals'),
    State('store-state', 'data'),
    State('store-trajectories', 'data'),
    prevent_initial_call=True
)


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

    app.run_server(
        host=config.APP_HOST,
        port=config.APP_PORT,
        debug=config.DEBUG
    )
