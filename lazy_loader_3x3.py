#!/usr/bin/env python3
"""
Lazy Loading 3x3 FOV Tile Navigator
Google Maps-style navigation for large microscopy datasets.
"""

import re
import csv
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import deque
import numpy as np
import tifffile as tf
from PIL import Image
from PyQt6.QtCore import Qt, QTimer, QPoint, QRect, pyqtSignal, QThread, QObject
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QSlider, QFileDialog, QSpinBox
)
from PyQt6.QtGui import QPixmap, QPainter, QImage, QWheelEvent
import ndv

# Pattern for acquisitions: manual_{fov}_{z}_Fluorescence_{wavelength}_nm_Ex.tiff
FPATTERN = re.compile(
    r"manual_(?P<f>\d+)_(?P<z>\d+)_Fluorescence_(?P<wavelength>\d+)_nm_Ex\.tiff?", re.IGNORECASE
)

@dataclass
class TileCoordinate:
    """Represents a position in the FOV grid"""
    fov_x: int
    fov_y: int
    z_level: int
    time: int
    channel: str
    
    def __hash__(self):
        return hash((self.fov_x, self.fov_y, self.z_level, self.time, self.channel))

class TileCache:
    """Multi-level cache system for tiles"""
    def __init__(self, max_tiles_l1=9, max_tiles_l2=16):
        self.l1_cache = {}  # Active tiles (3x3)
        self.l2_cache = {}  # Prefetch buffer
        self.max_l1 = max_tiles_l1
        self.max_l2 = max_tiles_l2
        self.access_order = deque()
        
    def get(self, coord: TileCoordinate) -> Optional[np.ndarray]:
        """Get tile from cache"""
        if coord in self.l1_cache:
            return self.l1_cache[coord]
        if coord in self.l2_cache:
            # Promote to L1
            self.put_l1(coord, self.l2_cache[coord])
            del self.l2_cache[coord]
            return self.l1_cache[coord]
        return None
    
    def put_l1(self, coord: TileCoordinate, data: np.ndarray):
        """Put tile in L1 cache"""
        self.l1_cache[coord] = data
        self.access_order.append(coord)
        
        # Evict if needed
        while len(self.l1_cache) > self.max_l1:
            # Move oldest to L2
            old_coord = self.access_order.popleft()
            if old_coord in self.l1_cache:
                self.l2_cache[old_coord] = self.l1_cache[old_coord]
                del self.l1_cache[old_coord]
                
        # Evict from L2 if needed
        while len(self.l2_cache) > self.max_l2:
            # Remove oldest from L2
            oldest = next(iter(self.l2_cache))
            del self.l2_cache[oldest]
    
    def put_l2(self, coord: TileCoordinate, data: np.ndarray):
        """Put tile in L2 cache"""
        if coord not in self.l1_cache:
            self.l2_cache[coord] = data

class FileIndex:
    """Maps tile coordinates to file paths"""
    def __init__(self):
        self.index = {}  # (time, fov_x, fov_y, z, channel) -> filepath
        self.grid_shape = (0, 0)  # (max_x, max_y)
        self.fov_to_grid = {}  # fov_id -> (x, y)
        self.grid_to_fov = {}  # (x, y) -> fov_id
        
    def build_from_coordinates(self, coord_file: Path, tiff_files: List[Path]):
        """Build index from coordinates.csv and file list"""
        # Read coordinates
        coordinates = {}
        with open(coord_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fov = int(row['fov'])
                x_mm = float(row['x (mm)'])
                y_mm = float(row['y (mm)'])
                coordinates[fov] = (x_mm, y_mm)
        
        # Convert to grid coordinates
        if coordinates:
            x_coords = [c[0] for c in coordinates.values()]
            y_coords = [c[1] for c in coordinates.values()]
            x_min, y_min = min(x_coords), min(y_coords)
            
            # Find unique x and y positions
            x_positions = sorted(set(x_coords))
            y_positions = sorted(set(y_coords))
            
            # Map stage coordinates to grid indices
            for fov, (x_mm, y_mm) in coordinates.items():
                # Find closest grid position
                x_idx = min(range(len(x_positions)), key=lambda i: abs(x_positions[i] - x_mm))
                y_idx = min(range(len(y_positions)), key=lambda i: abs(y_positions[i] - y_mm))
                
                self.fov_to_grid[fov] = (x_idx, y_idx)
                self.grid_to_fov[(x_idx, y_idx)] = fov
            
            self.grid_shape = (len(x_positions), len(y_positions))
        
        # Index files
        time = 0  # From parent directory
        for filepath in tiff_files:
            if m := FPATTERN.search(filepath.name):
                fov = int(m.group("f"))
                z = int(m.group("z"))
                wavelength = m.group("wavelength")
                channel = f"{wavelength}nm"
                
                if fov in self.fov_to_grid:
                    x, y = self.fov_to_grid[fov]
                    key = (time, x, y, z, channel)
                    self.index[key] = filepath
    
    def get_file(self, coord: TileCoordinate) -> Optional[Path]:
        """Get file path for coordinate"""
        key = (coord.time, coord.fov_x, coord.fov_y, coord.z_level, coord.channel)
        return self.index.get(key)

class TileLoader(QThread):
    """Asynchronous tile loader"""
    tile_loaded = pyqtSignal(TileCoordinate, np.ndarray)
    
    def __init__(self, file_index: FileIndex, cache: TileCache):
        super().__init__()
        self.file_index = file_index
        self.cache = cache
        self.load_queue = deque()
        self.running = True
        
    def queue_tile(self, coord: TileCoordinate, priority=False):
        """Add tile to load queue"""
        if priority:
            self.load_queue.appendleft(coord)
        else:
            self.load_queue.append(coord)
    
    def run(self):
        """Worker thread main loop"""
        while self.running:
            if self.load_queue:
                coord = self.load_queue.popleft()
                
                # Check if already cached
                if self.cache.get(coord) is not None:
                    continue
                
                # Load from disk
                filepath = self.file_index.get_file(coord)
                if filepath and filepath.exists():
                    try:
                        data = tf.imread(filepath)
                        self.tile_loaded.emit(coord, data)
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
            else:
                self.msleep(10)  # Short sleep when idle
    
    def stop(self):
        self.running = False
        self.wait()

class Viewport:
    """Manages the current 3x3 viewing area"""
    def __init__(self, center: TileCoordinate, grid_shape: Tuple[int, int]):
        self.center = center
        self.grid_shape = grid_shape
        
    def get_visible_tiles(self) -> List[TileCoordinate]:
        """Get the 3x3 grid centered on current position"""
        tiles = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x = self.center.fov_x + dx
                y = self.center.fov_y + dy
                
                if 0 <= x < self.grid_shape[0] and 0 <= y < self.grid_shape[1]:
                    tiles.append(TileCoordinate(
                        x, y, self.center.z_level,
                        self.center.time, self.center.channel
                    ))
        return tiles
    
    def get_prefetch_tiles(self) -> List[TileCoordinate]:
        """Get surrounding tiles for prefetching (5x5 grid minus 3x3)"""
        tiles = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) > 1 or abs(dy) > 1:  # Outside 3x3
                    x = self.center.fov_x + dx
                    y = self.center.fov_y + dy
                    
                    if 0 <= x < self.grid_shape[0] and 0 <= y < self.grid_shape[1]:
                        tiles.append(TileCoordinate(
                            x, y, self.center.z_level,
                            self.center.time, self.center.channel
                        ))
        return tiles

class NavigatorWidget(QWidget):
    """Main navigation widget"""
    def __init__(self):
        super().__init__()
        self.file_index = FileIndex()
        self.cache = TileCache()
        self.loader = TileLoader(self.file_index, self.cache)
        self.viewport = None
        self.tile_shape = None
        self.channels = []
        self.z_levels = 1
        self.loaded_tiles = {}
        self.timepoint_dir = None
        
        # Display settings
        self.tile_size = 256  # Display size for each tile
        self.contrast_min = 0
        self.contrast_max = 65535
        
        # Navigation
        self.pan_start = None
        self.view_offset = QPoint(0, 0)
        
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Controls
        controls = QHBoxLayout()
        
        controls.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        controls.addWidget(self.channel_combo)
        
        controls.addWidget(QLabel("Z:"))
        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_label = QLabel("0")
        controls.addWidget(self.z_slider)
        controls.addWidget(self.z_label)
        
        controls.addWidget(QLabel("Min:"))
        self.min_spin = QSpinBox()
        self.min_spin.setRange(0, 65535)
        self.min_spin.setValue(0)
        controls.addWidget(self.min_spin)
        
        controls.addWidget(QLabel("Max:"))
        self.max_spin = QSpinBox()
        self.max_spin.setRange(0, 65535)
        self.max_spin.setValue(65535)
        controls.addWidget(self.max_spin)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # Canvas
        self.canvas = QLabel()
        self.canvas.setMinimumSize(800, 600)
        self.canvas.setStyleSheet("background-color: black;")
        layout.addWidget(self.canvas)
        
        # Status
        self.status_label = QLabel("No data loaded")
        layout.addWidget(self.status_label)
        
    def _connect_signals(self):
        self.loader.tile_loaded.connect(self._on_tile_loaded)
        self.channel_combo.currentTextChanged.connect(self._on_channel_changed)
        self.z_slider.valueChanged.connect(self._on_z_changed)
        self.min_spin.valueChanged.connect(self._update_contrast)
        self.max_spin.valueChanged.connect(self._update_contrast)
        
    def load_acquisition(self, acquisition_dir: Path):
        """Load acquisition directory"""
        # Find timepoint directories
        timepoint_dirs = [d for d in acquisition_dir.iterdir() 
                         if d.is_dir() and d.name.isdigit()]
        
        if not timepoint_dirs:
            return
        
        self.timepoint_dir = sorted(timepoint_dirs)[0]
        coord_file = self.timepoint_dir / "coordinates.csv"
        
        if not coord_file.exists():
            return
        
        # Get all TIFF files
        tiff_files = list(self.timepoint_dir.glob("*.tif*"))
        
        # Build index
        self.file_index.build_from_coordinates(coord_file, tiff_files)
        
        # Get channels and z-levels
        channels = set()
        z_levels = set()
        for key in self.file_index.index.keys():
            channels.add(key[4])  # channel
            z_levels.add(key[3])  # z
        
        self.channels = sorted(channels)
        self.z_levels = len(z_levels)
        
        # Update UI
        self.channel_combo.clear()
        self.channel_combo.addItems(self.channels)
        
        self.z_slider.setRange(0, self.z_levels - 1)
        self.z_slider.setValue(self.z_levels // 2)
        
        # Get tile shape from first file
        first_file = next(iter(self.file_index.index.values()))
        sample = tf.imread(first_file)
        self.tile_shape = sample.shape
        
        # Initialize viewport at center
        center_x = self.file_index.grid_shape[0] // 2
        center_y = self.file_index.grid_shape[1] // 2
        
        self.viewport = Viewport(
            TileCoordinate(center_x, center_y, self.z_levels // 2, 0, self.channels[0]),
            self.file_index.grid_shape
        )
        
        # Start loader
        self.loader.start()
        
        # Initial load
        self._update_viewport()
        
    def _update_viewport(self):
        """Update viewport and trigger loading"""
        if not self.viewport:
            return
        
        # Clear loaded tiles for new position
        self.loaded_tiles.clear()
        
        # Queue visible tiles with high priority
        for tile in self.viewport.get_visible_tiles():
            self.loader.queue_tile(tile, priority=True)
        
        # Queue prefetch tiles
        for tile in self.viewport.get_prefetch_tiles():
            self.loader.queue_tile(tile, priority=False)
        
        self._render()
        
    def _on_tile_loaded(self, coord: TileCoordinate, data: np.ndarray):
        """Handle loaded tile"""
        self.cache.put_l1(coord, data)
        self.loaded_tiles[coord] = data
        self._render()
        
    def _render(self):
        """Render current viewport"""
        if not self.viewport:
            return
        
        # Create canvas
        canvas_size = self.canvas.size()
        image = QImage(canvas_size.width(), canvas_size.height(), 
                      QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.black)
        
        painter = QPainter(image)
        
        # Get visible tiles
        visible_tiles = self.viewport.get_visible_tiles()
        
        # Find bounds
        min_x = min(t.fov_x for t in visible_tiles)
        min_y = min(t.fov_y for t in visible_tiles)
        
        # Render each tile
        for tile_coord in visible_tiles:
            # Get tile data
            data = self.cache.get(tile_coord)
            if data is None:
                continue
            
            # Convert to 8-bit with contrast
            data_8bit = self._apply_contrast(data)
            
            # Convert to QImage
            h, w = data_8bit.shape
            bytes_per_line = w
            q_image = QImage(data_8bit.data, w, h, bytes_per_line,
                           QImage.Format.Format_Grayscale8)
            
            # Scale to display size
            scaled = q_image.scaled(self.tile_size, self.tile_size,
                                  Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
            
            # Calculate position
            grid_x = tile_coord.fov_x - min_x
            grid_y = tile_coord.fov_y - min_y
            
            x = grid_x * self.tile_size + self.view_offset.x() + canvas_size.width() // 2 - self.tile_size * 1.5
            y = grid_y * self.tile_size + self.view_offset.y() + canvas_size.height() // 2 - self.tile_size * 1.5
            
            painter.drawImage(int(x), int(y), scaled)
        
        painter.end()
        
        # Display
        self.canvas.setPixmap(QPixmap.fromImage(image))
        
        # Update status
        loaded = sum(1 for t in visible_tiles if self.cache.get(t) is not None)
        self.status_label.setText(
            f"Position: ({self.viewport.center.fov_x}, {self.viewport.center.fov_y}) | "
            f"Z: {self.viewport.center.z_level} | "
            f"Loaded: {loaded}/{len(visible_tiles)}"
        )
    
    def _apply_contrast(self, data: np.ndarray) -> np.ndarray:
        """Apply contrast adjustment"""
        # Clip and scale to 8-bit
        clipped = np.clip(data, self.contrast_min, self.contrast_max)
        if self.contrast_max > self.contrast_min:
            scaled = (clipped - self.contrast_min) / (self.contrast_max - self.contrast_min)
        else:
            scaled = np.zeros_like(clipped)
        return (scaled * 255).astype(np.uint8)
    
    def _update_contrast(self):
        """Update contrast settings"""
        self.contrast_min = self.min_spin.value()
        self.contrast_max = self.max_spin.value()
        self._render()
    
    def _on_channel_changed(self, channel: str):
        """Handle channel change"""
        if self.viewport and channel:
            self.viewport.center.channel = channel
            self._update_viewport()
    
    def _on_z_changed(self, z: int):
        """Handle z-level change"""
        if self.viewport:
            self.viewport.center.z_level = z
            self.z_label.setText(str(z))
            self._update_viewport()
    
    def mousePressEvent(self, event):
        """Start pan"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.pan_start = event.pos()
    
    def mouseMoveEvent(self, event):
        """Handle pan"""
        if self.pan_start:
            delta = event.pos() - self.pan_start
            self.view_offset += delta
            self.pan_start = event.pos()
            
            # Check if we need to move viewport
            if abs(self.view_offset.x()) > self.tile_size:
                dx = -self.view_offset.x() // self.tile_size
                self.view_offset.setX(self.view_offset.x() + dx * self.tile_size)
                new_x = max(0, min(self.file_index.grid_shape[0] - 1,
                                  self.viewport.center.fov_x + dx))
                if new_x != self.viewport.center.fov_x:
                    self.viewport.center.fov_x = new_x
                    self._update_viewport()
            
            if abs(self.view_offset.y()) > self.tile_size:
                dy = -self.view_offset.y() // self.tile_size
                self.view_offset.setY(self.view_offset.y() + dy * self.tile_size)
                new_y = max(0, min(self.file_index.grid_shape[1] - 1,
                                  self.viewport.center.fov_y + dy))
                if new_y != self.viewport.center.fov_y:
                    self.viewport.center.fov_y = new_y
                    self._update_viewport()
            
            self._render()
    
    def mouseReleaseEvent(self, event):
        """End pan"""
        self.pan_start = None
    
    def wheelEvent(self, event):
        """Handle zoom (placeholder)"""
        # For now, just adjust tile size slightly
        delta = event.angleDelta().y()
        if delta > 0:
            self.tile_size = min(512, int(self.tile_size * 1.1))
        else:
            self.tile_size = max(128, int(self.tile_size * 0.9))
        self._render()
    
    def keyPressEvent(self, event):
        """Handle keyboard navigation"""
        if not self.viewport:
            return
        
        dx, dy = 0, 0
        if event.key() == Qt.Key.Key_Left:
            dx = -1
        elif event.key() == Qt.Key.Key_Right:
            dx = 1
        elif event.key() == Qt.Key.Key_Up:
            dy = -1
        elif event.key() == Qt.Key.Key_Down:
            dy = 1
        
        if dx or dy:
            new_x = max(0, min(self.file_index.grid_shape[0] - 1,
                              self.viewport.center.fov_x + dx))
            new_y = max(0, min(self.file_index.grid_shape[1] - 1,
                              self.viewport.center.fov_y + dy))
            
            if new_x != self.viewport.center.fov_x or new_y != self.viewport.center.fov_y:
                self.viewport.center.fov_x = new_x
                self.viewport.center.fov_y = new_y
                self._update_viewport()

class LazyNavigatorWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lazy FOV Navigator")
        self.resize(1000, 800)
        
        self.navigator = NavigatorWidget()
        self.setCentralWidget(self.navigator)
        
        # Menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        open_action = file_menu.addAction("Open Acquisition")
        open_action.triggered.connect(self._open_acquisition)
        
    def _open_acquisition(self):
        """Open acquisition directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Acquisition Directory"
        )
        if dir_path:
            self.navigator.load_acquisition(Path(dir_path))
    
    def closeEvent(self, event):
        """Clean up on close"""
        self.navigator.loader.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = LazyNavigatorWindow()
    window.show()
    
    # Load from command line if provided
    if len(sys.argv) > 1:
        window.navigator.load_acquisition(Path(sys.argv[1]))
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()