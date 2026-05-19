"""
CASA-Net: Context-Aware Small Object Detection Network for UAV Aerial Images
"""

from casa_net.models.casa_net import CASAYOLOv26, build_casa_net_model

__all__ = [
    'CASAYOLOv26',
    'build_casa_net_model',
]

__version__ = '1.0.0'
