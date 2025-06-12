# Lazy FOV Navigator

Google Maps-style navigation for large microscopy datasets with lazy loading.

## Install
```bash
pip install -r requirements.txt
python create_desktop_shortcut.py
```

## Usage
```bash
python lazy_navigator.py [acquisition_dir]
```
Or use desktop shortcut.

## Features
- **3x3 tile viewport** with lazy loading
- **Smooth panning** with mouse drag
- **Keyboard navigation** (arrow keys)
- **Multi-level cache** (L1: active tiles, L2: prefetch)
- **Z-stack & channel switching**
- **Real-time contrast adjustment**

## Controls
- **Mouse drag**: Pan view
- **Arrow keys**: Move FOV by FOV
- **Mouse wheel**: Zoom in/out
- **Sliders**: Adjust Z-level and contrast
