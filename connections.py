"""
connections.py
==============
mesh.py içindeki import'u karşılamak için MediaPipe face-mesh bağlantı
setlerini tek noktadan dışa aktarır.

MediaPipe >= 0.10.x sürümlerinde bağlantılar objelere dönüştü ve 
bazı isimler (örn: FACEMESH_IRISES) ayrılarak değiştirildi.
Bu dosya onları otomatik olarak eski tuple (start, end) formatına çevirir.
"""

import mediapipe as mp

def _convert_connections(connections_list):
    if not connections_list:
        return frozenset()
    
    formatted = []
    for c in connections_list:
        if hasattr(c, 'start') and hasattr(c, 'end'):
            formatted.append((c.start, c.end))
        elif isinstance(c, tuple) or isinstance(c, list):
            formatted.append(tuple(c))
    return frozenset(formatted)

try:
    # Yeni MediaPipe (Tasks API - 0.10.x ve sonrası)
    from mediapipe.tasks.python.vision import face_landmarker
    _conn = face_landmarker.FaceLandmarksConnections
    FACEMESH_TESSELATION = _convert_connections(_conn.FACE_LANDMARKS_TESSELATION)
    FACEMESH_CONTOURS    = _convert_connections(_conn.FACE_LANDMARKS_CONTOURS)
    
    # Modern sürümde irises sol ve sağ olarak ayrılmıştır, birleştiriyoruz:
    _left_iris = _convert_connections(getattr(_conn, 'FACE_LANDMARKS_LEFT_IRIS', []))
    _right_iris = _convert_connections(getattr(_conn, 'FACE_LANDMARKS_RIGHT_IRIS', []))
    FACEMESH_IRISES = frozenset(list(_left_iris) + list(_right_iris))
    
except (ImportError, AttributeError):
    # Eski MediaPipe (Legacy Solutions API <= 0.10.15)
    _face_mesh = mp.solutions.face_mesh
    FACEMESH_TESSELATION = frozenset(_face_mesh.FACEMESH_TESSELATION)
    FACEMESH_CONTOURS    = frozenset(_face_mesh.FACEMESH_CONTOURS)
    FACEMESH_IRISES      = frozenset(_face_mesh.FACEMESH_IRISES)
