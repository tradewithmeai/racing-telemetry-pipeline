"""CLI utility for processing race track SVG files.

Usage:
    python process_track.py barber                    # Process Barber track
    python process_track.py --init spa --svg spa.svg  # Initialize new track
    python process_track.py --list                    # List available tracks
    python process_track.py barber --force            # Force re-process
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import yaml

from track_processing import Track

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
TRACKS_DIR = SCRIPT_DIR / 'tracks'


def init_track(track_name: str, svg_path: str):
    """Initialize a new track from SVG.

    Args:
        track_name: Name of track (e.g., 'watkins_glen')
        svg_path: Path to SVG file
    """
    track_dir = TRACKS_DIR / track_name
    svg_source = Path(svg_path)

    if not svg_source.exists():
        logger.error(f"SVG file not found: {svg_path}")
        sys.exit(1)

    if track_dir.exists():
        logger.error(f"Track directory already exists: {track_dir}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

    # Create directory
    track_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created track directory: {track_dir}")

    # Copy SVG
    svg_dest = track_dir / svg_source.name
    shutil.copy(svg_source, svg_dest)
    logger.info(f"Copied SVG: {svg_dest}")

    # Create default config
    config = {
        'track': {
            'name': track_name.replace('_', ' ').title(),
            'location': '',
            'country': '',
        },
        'geometry': {
            'length_m': 3830.0,  # Update with actual value
            'width_m': 13.72,    # Update with actual value
            'direction': 'clockwise',
            'turns': None,
            'elevation_change_m': None,
        },
        'svg': {
            'source': svg_source.name,
            'auto_detect': True,
            'override_length': None,  # Set to override auto-detection
        },
        'ribbons': {
            'per_side': 5,
            'max_offset_m': 6.86,  # Half track width
            'join_style': 'round',
            'quad_segs': 12,
        },
        'car_assignments': {},  # Auto-assign if empty
        'processing': {
            'svg_samples_per_seg': 4000,
            'centerline_points': 5000,
            'smoothing': 0.0,
        },
    }

    config_path = track_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Created config: {config_path}")

    print(f"\n{'='*60}")
    print(f"Track '{track_name}' initialized!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Edit {config_path}")
    print(f"   - Set geometry.length_m to official track length")
    print(f"   - Set geometry.width_m to track width")
    print(f"   - Update track metadata")
    print(f"2. Run: python process_track.py {track_name}")
    print(f"3. Update config.py: CURRENT_TRACK = '{track_name}'")
    print(f"4. Restart dashboard")


def process_track(track_name: str, force: bool = False):
    """Process SVG → centerline → ribbons for a track.

    Args:
        track_name: Name of track directory
        force: Force re-processing even if files exist
    """
    track_dir = TRACKS_DIR / track_name

    if not track_dir.exists():
        logger.error(f"Track directory not found: {track_dir}")
        logger.info(f"Available tracks: {list_tracks_internal()}")
        sys.exit(1)

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing track: {track_name}")
    logger.info(f"{'='*60}\n")

    try:
        track = Track(track_dir)

        # Force delete existing files if requested
        if force:
            for fname in ['track_centerline.npy', 'track_ribbons.json', 'track_meta.json']:
                fpath = track_dir / fname
                if fpath.exists():
                    fpath.unlink()
                    logger.info(f"Deleted existing: {fname}")

        # Process (auto-generates if missing)
        processed = track.ensure_processed()

        if processed:
            logger.info(f"\n{'='*60}")
            logger.info(f"Track '{track_name}' processed successfully!")
            logger.info(f"{'='*60}")
        else:
            logger.info(f"\n{'='*60}")
            logger.info(f"Track '{track_name}' already processed")
            logger.info(f"Use --force to re-process")
            logger.info(f"{'='*60}")

        # Show summary
        track.load_geometry()
        info = track.get_track_info()

        print(f"\nTrack Information:")
        print(f"  Name: {info['name']}")
        print(f"  Location: {info['location'] or 'N/A'}")
        print(f"  Length: {info['length_m']:.2f}m ({info['length_m']/1000:.2f}km)")
        print(f"  Width: {info['width_m']:.2f}m")
        print(f"  Direction: {info['direction']}")
        print(f"  Turns: {info['turns'] or 'N/A'}")

        print(f"\nGenerated Files:")
        print(f"  Centerline: {track.centerline.shape[0]} points")
        print(f"  Ribbons: {len(track.ribbons['ribbons'])} total")
        print(f"    - {track.ribbons['meta']['ribbons_per_side']} per side")
        print(f"    - Max offset: ±{track.ribbons['meta']['max_half_width_m']:.2f}m")
        print(f"    - Spacing: {track.ribbons['meta']['spacing_m']:.2f}m")

    except Exception as e:
        logger.error(f"Failed to process track: {e}", exc_info=True)
        sys.exit(1)


def list_tracks_internal():
    """Get list of available tracks."""
    if not TRACKS_DIR.exists():
        return []

    return [d.name for d in TRACKS_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]


def list_tracks():
    """List all available tracks."""
    tracks = list_tracks_internal()

    if not tracks:
        print("No tracks found.")
        print(f"Track directory: {TRACKS_DIR}")
        print("\nUse --init to create a new track.")
        return

    print(f"\n{'='*60}")
    print(f"Available Tracks ({len(tracks)})")
    print(f"{'='*60}\n")

    for track_name in sorted(tracks):
        track_dir = TRACKS_DIR / track_name
        config_path = track_dir / 'config.yaml'

        # Load basic info
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            name = config.get('track', {}).get('name', track_name)
            length = config.get('geometry', {}).get('length_m', 0)

            # Check if processed
            centerline_exists = (track_dir / 'track_centerline.npy').exists()
            status = "✓ Processed" if centerline_exists else "○ Not processed"

            print(f"  {track_name:<20} {name:<30} {length/1000:.2f}km  {status}")

        except Exception as e:
            print(f"  {track_name:<20} (Error loading config: {e})")

    print()


def validate_track(track_name: str):
    """Validate track geometry and configuration.

    Args:
        track_name: Name of track to validate
    """
    track_dir = TRACKS_DIR / track_name

    if not track_dir.exists():
        logger.error(f"Track directory not found: {track_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Validating track: {track_name}")
    print(f"{'='*60}\n")

    issues = []

    # Check config
    config_path = track_dir / 'config.yaml'
    if not config_path.exists():
        issues.append("❌ config.yaml missing")
    else:
        print("✓ config.yaml found")

    # Check SVG
    try:
        track = Track(track_dir)
        svg_path = track_dir / track.config['svg']['source']
        if not svg_path.exists():
            issues.append(f"❌ SVG not found: {svg_path}")
        else:
            print(f"✓ SVG found: {svg_path.name}")
    except Exception as e:
        issues.append(f"❌ Config load error: {e}")

    # Check processed files
    for fname in ['track_centerline.npy', 'track_ribbons.json', 'track_meta.json']:
        fpath = track_dir / fname
        if fpath.exists():
            print(f"✓ {fname} exists")
        else:
            print(f"○ {fname} missing (will be auto-generated)")

    if issues:
        print(f"\n{'='*60}")
        print("Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        sys.exit(1)
    else:
        print(f"\n{'='*60}")
        print("✓ Track validation passed!")


def main():
    parser = argparse.ArgumentParser(
        description='Process race track SVG files',
        epilog='Examples:\n'
               '  python process_track.py barber\n'
               '  python process_track.py --init spa --svg spa.svg\n'
               '  python process_track.py --list',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('track_name', nargs='?', help='Name of track to process')
    parser.add_argument('--init', metavar='NAME', help='Initialize new track')
    parser.add_argument('--svg', metavar='PATH', help='SVG file path (with --init)')
    parser.add_argument('--list', action='store_true', help='List available tracks')
    parser.add_argument('--validate', action='store_true', help='Validate track configuration')
    parser.add_argument('--force', action='store_true', help='Force re-processing')

    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_tracks()
        return

    # Handle --init
    if args.init:
        if not args.svg:
            logger.error("--svg required with --init")
            sys.exit(1)
        init_track(args.init, args.svg)
        return

    # Handle track processing
    if not args.track_name:
        parser.print_help()
        sys.exit(1)

    if args.validate:
        validate_track(args.track_name)
    else:
        process_track(args.track_name, force=args.force)


if __name__ == '__main__':
    main()
