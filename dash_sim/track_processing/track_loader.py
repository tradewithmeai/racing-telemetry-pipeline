"""Track loader with auto-processing capabilities."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import yaml

from .svg_extractor import extract_centerline
from .ribbon_generator import generate_ribbons

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    'track': {
        'name': 'Unknown Track',
        'location': '',
        'country': '',
    },
    'geometry': {
        'length_m': 3830.0,
        'width_m': 13.72,
        'direction': 'clockwise',
        'turns': None,
        'elevation_change_m': None,
    },
    'svg': {
        'source': 'track.svg',
        'auto_detect': True,
        'override_length': None,
    },
    'ribbons': {
        'per_side': 5,
        'max_offset_m': 6.86,
        'join_style': 'round',
        'quad_segs': 12,
    },
    'car_assignments': {},
    'processing': {
        'svg_samples_per_seg': 4000,
        'centerline_points': 5000,
        'smoothing': 0.0,
    },
}


class Track:
    """Represents a race track with geometry and configuration.

    Handles loading, validation, and auto-generation of track data.
    """

    def __init__(self, track_dir: Path):
        """Initialize track from directory.

        Args:
            track_dir: Path to track directory (e.g., tracks/barber/)
        """
        self.track_dir = Path(track_dir)
        self.config = self._load_config()
        self.centerline: Optional[np.ndarray] = None
        self.ribbons: Optional[Dict] = None
        self.meta: Optional[Dict] = None

    def _load_config(self) -> Dict:
        """Load and validate config.yaml from track directory.

        Returns:
            Configuration dict with defaults applied
        """
        config_path = self.track_dir / 'config.yaml'

        if not config_path.exists():
            logger.warning(f"No config.yaml found in {self.track_dir}, using defaults")
            return DEFAULT_CONFIG.copy()

        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)

            # Merge with defaults (deep update)
            config = DEFAULT_CONFIG.copy()
            for section, values in user_config.items():
                if section in config and isinstance(config[section], dict):
                    config[section].update(values)
                else:
                    config[section] = values

            logger.info(f"Loaded config from {config_path}")
            return config

        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            logger.warning("Using default configuration")
            return DEFAULT_CONFIG.copy()

    def _get_file_paths(self) -> Dict[str, Path]:
        """Get paths to track files.

        Returns:
            dict with paths to svg, centerline, ribbons, meta files
        """
        svg_name = self.config['svg']['source']

        return {
            'svg': self.track_dir / svg_name,
            'centerline': self.track_dir / 'track_centerline.npy',
            'ribbons': self.track_dir / 'track_ribbons.json',
            'meta': self.track_dir / 'track_meta.json',
        }

    def ensure_processed(self) -> bool:
        """Auto-generate centerline/ribbons if missing.

        Returns:
            True if processing occurred, False if files already exist
        """
        paths = self._get_file_paths()

        # Check if outputs exist
        if paths['centerline'].exists() and paths['ribbons'].exists():
            logger.debug(f"Track files already exist for {self.track_dir.name}")
            return False

        logger.info(f"Processing track: {self.track_dir.name}")

        # Check SVG exists
        if not paths['svg'].exists():
            raise FileNotFoundError(
                f"SVG source not found: {paths['svg']}\n"
                f"Expected: {self.config['svg']['source']}"
            )

        # Extract centerline
        target_length = (
            self.config['svg']['override_length']
            or self.config['geometry']['length_m']
        )

        centerline, meta = extract_centerline(
            paths['svg'],
            target_length_m=target_length,
            samples=self.config['processing']['centerline_points'],
            smoothing=self.config['processing']['smoothing'],
            ensure_direction_val=self.config['geometry']['direction'],
            samples_per_seg=self.config['processing']['svg_samples_per_seg'],
        )

        # Save centerline
        np.save(paths['centerline'], centerline)
        logger.info(f"Saved centerline: {paths['centerline']}")

        # Generate ribbons
        ribbons = generate_ribbons(
            centerline,
            ribbons_per_side=self.config['ribbons']['per_side'],
            max_half_width_m=self.config['ribbons']['max_offset_m'],
            join_style=self.config['ribbons']['join_style'],
            quad_segs=self.config['ribbons']['quad_segs'],
        )

        # Save ribbons
        with open(paths['ribbons'], 'w') as f:
            json.dump(ribbons, f, indent=2)
        logger.info(f"Saved ribbons: {paths['ribbons']}")

        # Save metadata
        with open(paths['meta'], 'w') as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved metadata: {paths['meta']}")

        return True

    def load_geometry(self):
        """Load centerline, ribbons, and metadata from files.

        Lazy-loads on first access. Auto-generates if missing.
        """
        if self.centerline is not None and self.ribbons is not None:
            return  # Already loaded

        # Ensure files exist
        self.ensure_processed()

        paths = self._get_file_paths()

        # Load centerline
        self.centerline = np.load(paths['centerline'])
        logger.info(f"Loaded centerline: {self.centerline.shape[0]} points")

        # Load ribbons
        with open(paths['ribbons'], 'r') as f:
            self.ribbons = json.load(f)
        logger.info(f"Loaded {len(self.ribbons['ribbons'])} ribbons")

        # Load metadata
        if paths['meta'].exists():
            with open(paths['meta'], 'r') as f:
                self.meta = json.load(f)

    def get_ribbon_for_car(self, car_id: str) -> str:
        """Get ribbon assignment for a car.

        Args:
            car_id: Car chassis ID (e.g., "010")

        Returns:
            Ribbon name (e.g., "left_1.37m")
        """
        assignments = self.config.get('car_assignments', {})

        if car_id in assignments:
            return assignments[car_id]

        # Auto-assign based on car number
        # Distribute cars across ribbons
        car_num = int(car_id) if car_id.isdigit() else hash(car_id)
        ribbons_per_side = self.config['ribbons']['per_side']
        total_ribbons = 2 * ribbons_per_side + 1

        ribbon_idx = car_num % total_ribbons
        spacing = self.config['ribbons']['max_offset_m'] / ribbons_per_side

        # Map index to ribbon name
        if ribbon_idx == 0:
            return 'center'
        elif ribbon_idx <= ribbons_per_side:
            offset = ribbon_idx * spacing
            return f'left_{offset:.2f}m'
        else:
            offset = (ribbon_idx - ribbons_per_side) * spacing
            return f'right_{offset:.2f}m'

    def get_track_info(self) -> Dict:
        """Get track information summary.

        Returns:
            dict with track name, length, width, etc.
        """
        return {
            'name': self.config['track']['name'],
            'location': self.config['track']['location'],
            'country': self.config['track']['country'],
            'length_m': self.config['geometry']['length_m'],
            'width_m': self.config['geometry']['width_m'],
            'direction': self.config['geometry']['direction'],
            'turns': self.config['geometry'].get('turns'),
            'elevation_change_m': self.config['geometry'].get('elevation_change_m'),
        }
