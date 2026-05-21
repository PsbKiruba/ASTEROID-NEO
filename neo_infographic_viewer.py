#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
    from PySide6.QtGui import QAction, QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPalette, QPen, QRadialGradient
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSplitter,
        QStyleFactory,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
except Exception:
    try:
        from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal as Signal
        from PyQt6.QtGui import QAction, QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPalette, QPen, QRadialGradient
        from PyQt6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QSizePolicy,
            QSlider,
            QSplitter,
            QStyleFactory,
            QToolBar,
            QVBoxLayout,
            QWidget,
        )
    except Exception as exc:  # pragma: no cover - import-time user guidance
        raise SystemExit(
            "This viewer needs a Qt6 Python binding. Install one with "
            "`python -m pip install PySide6` or `python -m pip install PyQt6`. "
            f"Import error: {exc}"
        ) from exc


AU_KM = 149_597_870.7
DEFAULT_BUNDLE = Path("outputs/predictor_3x_samples")


@dataclass(frozen=True)
class Anchor:
    label: str
    jd: float
    cad_distance_km: float
    integrated_minus_cad_km: float
    horizons_minus_cad_km: float
    sample_offset_hours: float
    nearest_index: int


@dataclass(frozen=True)
class RunData:
    bundle: Path
    report: dict[str, Any]
    jd: np.ndarray
    calendar: list[str]
    horizons_dist_au: np.ndarray
    integrated_dist_au: np.ndarray
    residual_km: np.ndarray
    pos_au: np.ndarray
    vel_au_d: np.ndarray
    gi_log: np.ndarray
    oi_log: np.ndarray
    anchors: list[Anchor]
    earth_fixed_au: np.ndarray
    close_index: int

    @property
    def speed_km_s(self) -> np.ndarray:
        return np.linalg.norm(self.vel_au_d, axis=1) * AU_KM / 86_400.0

    @property
    def range_rate_km_s(self) -> np.ndarray:
        seconds = self.jd * 86_400.0
        return np.gradient(self.integrated_dist_au * AU_KM, seconds)


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _short_date(text: str) -> str:
    parts = str(text).replace("A.D.", "").strip().split()
    return " ".join(parts[:2]) if len(parts) >= 2 else str(text)


def _fmt_km(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000:
        return f"{sign}{value / 1_000_000:.2f}M km"
    if value >= 10_000:
        return f"{sign}{value / 1_000:.1f}k km"
    if value >= 100:
        return f"{sign}{value:,.0f} km"
    return f"{sign}{value:.1f} km"


def _fmt_au(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.6f} au"


def _fmt_gate(value: Any) -> str:
    if value is True:
        return "accepted"
    if value is False:
        return "rejected"
    return "n/a"


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_run(bundle: Path) -> RunData:
    bundle = bundle.expanduser().resolve()
    report_path = bundle / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing required file: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    rows = _load_csv_rows(bundle / "table_dynamical_integrator_timeseries.csv")
    if len(rows) < 2:
        raise ValueError("Timeseries needs at least two samples for animation")

    jd = np.asarray([_to_float(row["jd_tdb"]) for row in rows], dtype=float)
    calendar = [row["calendar_tdb"] for row in rows]
    horizons_dist_au = np.asarray([_to_float(row["horizons_geocentric_distance_au"]) for row in rows], dtype=float)
    integrated_dist_au = np.asarray([_to_float(row["integrated_geocentric_distance_au"]) for row in rows], dtype=float)
    residual_km = np.asarray([_to_float(row["integrated_minus_horizons_km"]) for row in rows], dtype=float)
    pos_au = np.asarray(
        [
            [
                _to_float(row["integrated_helio_x_au"]),
                _to_float(row["integrated_helio_y_au"]),
                _to_float(row["integrated_helio_z_au"]),
            ]
            for row in rows
        ],
        dtype=float,
    )
    vel_au_d = np.asarray(
        [
            [
                _to_float(row["integrated_helio_vx_au_d"]),
                _to_float(row["integrated_helio_vy_au_d"]),
                _to_float(row["integrated_helio_vz_au_d"]),
            ]
            for row in rows
        ],
        dtype=float,
    )
    gi_log = np.asarray([_to_float(row["gi_log10_abs"]) for row in rows], dtype=float)
    oi_log = np.asarray([_to_float(row["oi_log10_abs"]) for row in rows], dtype=float)

    anchors: list[Anchor] = []
    anchor_path = bundle / "table_dynamical_integrator_anchor_validation.csv"
    if anchor_path.exists():
        for row in _load_csv_rows(anchor_path):
            anchor_jd = _to_float(row.get("cad_jd_tdb"))
            nearest_index = int(np.argmin(np.abs(jd - anchor_jd))) if math.isfinite(anchor_jd) else 0
            anchors.append(
                Anchor(
                    label=str(row.get("cad_date_tdb") or "CAD anchor"),
                    jd=anchor_jd,
                    cad_distance_km=_to_float(row.get("cad_distance_km")),
                    integrated_minus_cad_km=_to_float(row.get("integrated_minus_cad_km")),
                    horizons_minus_cad_km=_to_float(row.get("horizons_interpolated_minus_cad_km")),
                    sample_offset_hours=_to_float(row.get("nearest_sample_offset_hours")),
                    nearest_index=nearest_index,
                )
            )

    if anchors:
        close_anchor = min(anchors, key=lambda item: item.cad_distance_km)
        close_index = int(close_anchor.nearest_index)
    else:
        close_index = int(np.argmin(integrated_dist_au))
    earth_fixed_au = np.asarray(pos_au[close_index], dtype=float)

    return RunData(
        bundle=bundle,
        report=report,
        jd=jd,
        calendar=calendar,
        horizons_dist_au=horizons_dist_au,
        integrated_dist_au=integrated_dist_au,
        residual_km=residual_km,
        pos_au=pos_au,
        vel_au_d=vel_au_d,
        gi_log=gi_log,
        oi_log=oi_log,
        anchors=anchors,
        earth_fixed_au=earth_fixed_au,
        close_index=close_index,
    )


class StatCard(QFrame):
    def __init__(self, title: str, value: str = "n/a", accent: str = "#71d3ff") -> None:
        super().__init__()
        self.title = QLabel(title)
        self.value = QLabel(value)
        self.value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.title.setObjectName("cardTitle")
        self.value.setObjectName("cardValue")
        self.setObjectName("statCard")
        self.setStyleSheet(
            f"""
            QFrame#statCard {{
                border: 1px solid rgba(255, 255, 255, 32);
                border-left: 3px solid {accent};
                background: rgba(11, 19, 32, 178);
                border-radius: 8px;
            }}
            QLabel#cardTitle {{
                color: rgba(216, 229, 242, 190);
                font-size: 11px;
                font-weight: 600;
            }}
            QLabel#cardValue {{
                color: #f5f8ff;
                font-size: 18px;
                font-weight: 700;
            }}
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 9)
        layout.setSpacing(2)
        layout.addWidget(self.title)
        layout.addWidget(self.value)

    def set_value(self, value: str) -> None:
        self.value.setText(value)


class TrajectoryCanvas(QWidget):
    indexChanged = Signal(int)

    def __init__(self, data: RunData) -> None:
        super().__init__()
        self.data = data
        self.index = 0
        self.yaw = -0.72
        self.pitch = 0.52
        self.zoom = 0.92
        self.auto_rotate = False
        self.view_mode = "solar"
        self.trail_points = 900
        self.max_orbit_draw_points = 1600
        self.max_trail_draw_points = 420
        self.full_draw_idx = np.linspace(
            0,
            len(data.pos_au) - 1,
            min(self.max_orbit_draw_points, len(data.pos_au)),
        ).astype(int)
        close_start = max(0, data.close_index - 1800)
        close_end = min(len(data.pos_au) - 1, data.close_index + 1800)
        self.close_draw_idx = np.linspace(close_start, close_end, min(900, close_end - close_start + 1)).astype(int)
        rng = np.random.default_rng(20290413)
        self.stars = np.column_stack(
            [
                rng.random(220),
                rng.random(220),
                rng.uniform(0.45, 1.9, 220),
                rng.uniform(34, 150, 220),
                rng.uniform(0.0, math.tau, 220),
            ]
        )
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._tick)
        self.step = max(1, len(data.jd) // 700)
        self._drag_pos: QPointF | None = None
        self.setMinimumSize(760, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def play(self) -> None:
        self.timer.start()

    def pause(self) -> None:
        self.timer.stop()

    def set_running(self, running: bool) -> None:
        self.play() if running else self.pause()

    def set_index(self, value: int) -> None:
        value = int(max(0, min(len(self.data.jd) - 1, value)))
        if value != self.index:
            self.index = value
            self.indexChanged.emit(value)
            self.update()

    def set_speed(self, value: int) -> None:
        self.step = max(1, int(value))

    def set_trail(self, value: int) -> None:
        self.trail_points = max(60, int(value))
        self.update()

    def set_auto_rotate(self, enabled: bool) -> None:
        self.auto_rotate = bool(enabled)

    def set_view_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        self.view_mode = "encounter" if "2029" in normalized or "encounter" in normalized else "solar"
        self.zoom = 1.18 if self.view_mode == "encounter" else 0.92
        self.update()

    def _tick(self) -> None:
        if self.auto_rotate:
            self.yaw += 0.002
        next_index = self.index + self.step
        if next_index >= len(self.data.jd):
            next_index = 0
        self.set_index(next_index)

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.position()

    def mouseMoveEvent(self, event: Any) -> None:
        if self._drag_pos is None:
            return
        pos = event.position()
        delta = pos - self._drag_pos
        self._drag_pos = pos
        self.yaw += float(delta.x()) * 0.009
        self.pitch = max(-1.25, min(1.25, self.pitch + float(delta.y()) * 0.007))
        self.update()

    def mouseReleaseEvent(self, event: Any) -> None:
        self._drag_pos = None

    def wheelEvent(self, event: Any) -> None:
        self.zoom = max(0.35, min(8.0, self.zoom * (1.0 + event.angleDelta().y() / 1800.0)))
        self.update()

    def _rotation_matrix(self) -> np.ndarray:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        ry = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        return rx @ ry

    def _encounter_active(self) -> bool:
        return self.view_mode == "encounter"

    def _scene_center_radius(self) -> tuple[np.ndarray, float]:
        if self._encounter_active():
            return np.asarray(self.data.earth_fixed_au, dtype=float), 0.026
        full_radius = max(float(np.nanmax(np.linalg.norm(self.data.pos_au, axis=1))), 1e-6)
        return np.zeros(3, dtype=float), full_radius

    def _project(self, pts: np.ndarray, rect: QRectF) -> tuple[np.ndarray, np.ndarray]:
        center, scene_radius = self._scene_center_radius()
        centered = pts - center
        rot = centered @ self._rotation_matrix().T
        scale = min(rect.width(), rect.height()) * 0.38 * self.zoom / scene_radius
        depth = 4.0 + rot[:, 2]
        perspective = 4.0 / np.maximum(depth, 0.45)
        x = rect.center().x() + rot[:, 0] * scale * perspective
        y = rect.center().y() - rot[:, 1] * scale * perspective
        return np.column_stack([x, y]), rot[:, 2]

    def _draw_polyline(self, painter: QPainter, points: np.ndarray, color: QColor, width: float, alpha: int = 255) -> None:
        if len(points) < 2:
            return
        color = QColor(color)
        color.setAlpha(alpha)
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        path = QPainterPath(QPointF(float(points[0, 0]), float(points[0, 1])))
        for x, y in points[1:]:
            path.lineTo(float(x), float(y))
        painter.drawPath(path)

    def _draw_chunked_trail(self, painter: QPainter, points: np.ndarray) -> None:
        if len(points) < 2:
            return
        chunks = np.array_split(points, min(7, max(1, len(points) // 24)))
        for i, chunk in enumerate(chunks):
            if len(chunk) < 2:
                continue
            t = (i + 1) / len(chunks)
            color = QColor.fromRgbF(0.18 + 0.48 * t, 0.68 + 0.23 * t, 1.0, 0.22 + 0.58 * t)
            self._draw_polyline(painter, chunk, color, 1.0 + 2.2 * t, color.alpha())

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(14, 14, -14, -14)
        self._paint_background(painter, rect)

        full_pts, _ = self._project(self.data.pos_au[self.full_draw_idx], rect)
        self._draw_orbit_halo(painter, rect)
        self._draw_polyline(painter, full_pts, QColor("#315b88"), 1.1, 112)

        start = max(0, self.index - self.trail_points)
        trail_idx = np.linspace(start, self.index, min(self.max_trail_draw_points, self.index - start + 1)).astype(int)
        if len(trail_idx) > 1:
            trail_pts, _ = self._project(self.data.pos_au[trail_idx], rect)
            self._draw_chunked_trail(painter, trail_pts)

        current_pt, _ = self._project(self.data.pos_au[[self.index]], rect)
        current = QPointF(float(current_pt[0, 0]), float(current_pt[0, 1]))
        self._draw_sun(painter, rect)
        self._draw_anchors(painter, rect)
        self._draw_current_body(painter, current)
        self._draw_hud(painter, rect)

    def _paint_background(self, painter: QPainter, rect: QRectF) -> None:
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, QColor("#050814"))
        gradient.setColorAt(0.45, QColor("#081625"))
        gradient.setColorAt(1.0, QColor("#12091a"))
        painter.fillRect(self.rect(), gradient)

        painter.setPen(QPen(QColor(255, 255, 255, 24), 1))
        center = rect.center()
        for r in np.linspace(0.18, 0.88, 5):
            radius = min(rect.width(), rect.height()) * float(r) * 0.5
            painter.drawEllipse(center, radius, radius * 0.58)
        painter.setPen(QPen(QColor("#315b88"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(rect.left(), center.y()), QPointF(rect.right(), center.y()))
        painter.drawLine(QPointF(center.x(), rect.top()), QPointF(center.x(), rect.bottom()))

    def _draw_starfield(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        drift_x = math.sin(self.yaw) * 18.0
        drift_y = math.sin(self.pitch) * 12.0
        for sx, sy, radius, alpha, phase in self.stars:
            x = rect.left() + float(sx) * rect.width() + drift_x * (float(radius) - 1.0)
            y = rect.top() + float(sy) * rect.height() + drift_y * (float(radius) - 1.0)
            color = QColor(210, 232, 255, int(float(alpha) * 0.68))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(x, y), float(radius), float(radius))

    def _draw_orbit_halo(self, painter: QPainter, rect: QRectF) -> None:
        center = rect.center()
        halo = QRadialGradient(center, min(rect.width(), rect.height()) * 0.48)
        halo.setColorAt(0.0, QColor(33, 148, 255, 18))
        halo.setColorAt(0.55, QColor(56, 103, 180, 7))
        halo.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(QBrush(halo))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, rect.width() * 0.42, rect.height() * 0.36)

    def _draw_close_corridor(self, painter: QPainter, rect: QRectF) -> None:
        corridor_pts, _ = self._project(self.data.pos_au[self.close_draw_idx], rect)
        active = self._encounter_active()
        self._draw_polyline(painter, corridor_pts, QColor("#7df9ff"), 7.0 if active else 2.4, 150 if active else 42)
        self._draw_polyline(painter, corridor_pts, QColor("#fff3b0"), 1.8 if active else 0.9, 210 if active else 64)
        if not active:
            return
        for idx in np.linspace(0, len(corridor_pts) - 1, 18).astype(int):
            point = QPointF(float(corridor_pts[idx, 0]), float(corridor_pts[idx, 1]))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(125, 249, 255, 34))
            painter.drawEllipse(point, 10, 10)

    def _draw_sun(self, painter: QPainter, rect: QRectF) -> None:
        sun_pt, _ = self._project(np.asarray([[0.0, 0.0, 0.0]], dtype=float), rect)
        sun = QPointF(float(sun_pt[0, 0]), float(sun_pt[0, 1]))
        grad = QRadialGradient(sun, 28)
        grad.setColorAt(0.0, QColor("#fff2a4"))
        grad.setColorAt(0.38, QColor("#ffab40"))
        grad.setColorAt(1.0, QColor(255, 157, 64, 0))
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(sun, 30, 30)
        painter.setBrush(QColor("#ffd166"))
        painter.drawEllipse(sun, 6, 6)

    def _draw_earth(self, painter: QPainter, rect: QRectF) -> None:
        earth_pt, _ = self._project(np.asarray([self.data.earth_fixed_au], dtype=float), rect)
        earth = QPointF(float(earth_pt[0, 0]), float(earth_pt[0, 1]))
        active = self._encounter_active()
        radius = 14.0 if active else 8.0
        halo = QRadialGradient(earth, radius * 4.0)
        halo.setColorAt(0.0, QColor(113, 211, 255, 118 if active else 64))
        halo.setColorAt(0.35, QColor(64, 132, 255, 40 if active else 18))
        halo.setColorAt(1.0, QColor(64, 132, 255, 0))
        painter.setBrush(QBrush(halo))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(earth, radius * 4.0, radius * 4.0)
        painter.setBrush(QColor("#3da9fc"))
        painter.setPen(QPen(QColor("#d9f2ff"), 1.2))
        painter.drawEllipse(earth, radius, radius)
        for i, (ring, color) in enumerate([(42.0, "#7df9ff"), (86.0, "#5fb3ff"), (138.0, "#fff3b0")]):
            alpha = int((78 if active else 20) / (i + 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(color).lighter(105), 1.1 if active else 0.8))
            pen_color = painter.pen().color()
            pen_color.setAlpha(alpha)
            pen = painter.pen()
            pen.setColor(pen_color)
            pen.setStyle(Qt.PenStyle.DashLine if i == 2 else Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawEllipse(earth, ring * (1.65 if active else 0.75), ring * (0.95 if active else 0.42))
        painter.setPen(QPen(QColor("#d9f2ff"), 1))
        painter.setFont(QFont("Helvetica Neue", 9, QFont.Weight.DemiBold))
        painter.drawText(earth + QPointF(12, -10), "Earth fixed")
        if active:
            painter.setPen(QColor(235, 245, 255, 145))
            painter.setFont(QFont("Helvetica Neue", 8, QFont.Weight.DemiBold))
            painter.drawText(earth + QPointF(22, 20), "encounter rings")

    def _draw_current_body(self, painter: QPainter, point: QPointF) -> None:
        grad = QRadialGradient(point, 28)
        grad.setColorAt(0.0, QColor("#ffffff"))
        grad.setColorAt(0.22, QColor("#7df9ff"))
        grad.setColorAt(1.0, QColor(125, 249, 255, 0))
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(point, 34, 34)
        painter.setBrush(QColor("#eaffff"))
        painter.drawEllipse(point, 5.5, 5.5)

    def _draw_velocity_vector(self, painter: QPainter, rect: QRectF) -> None:
        p0 = np.asarray(self.data.pos_au[self.index], dtype=float)
        v = np.asarray(self.data.vel_au_d[self.index], dtype=float)
        v_norm = max(float(np.linalg.norm(v)), 1e-12)
        direction = v / v_norm
        scene_radius = self._scene_center_radius()[1]
        p1 = p0 + direction * scene_radius * 0.18
        pts, _ = self._project(np.asarray([p0, p1], dtype=float), rect)
        a = QPointF(float(pts[0, 0]), float(pts[0, 1]))
        b = QPointF(float(pts[1, 0]), float(pts[1, 1]))
        painter.setPen(QPen(QColor(255, 243, 176, 210), 2.1))
        painter.drawLine(a, b)
        dx = b.x() - a.x()
        dy = b.y() - a.y()
        length = max(math.hypot(dx, dy), 1.0)
        ux, uy = dx / length, dy / length
        left = QPointF(b.x() - 12 * ux - 6 * uy, b.y() - 12 * uy + 6 * ux)
        right = QPointF(b.x() - 12 * ux + 6 * uy, b.y() - 12 * uy - 6 * ux)
        painter.drawLine(b, left)
        painter.drawLine(b, right)
        painter.setFont(QFont("Helvetica Neue", 8, QFont.Weight.DemiBold))
        painter.drawText(b + QPointF(7, -7), f"v {float(self.data.speed_km_s[self.index]):.2f} km/s")

    def _draw_probe_particles(self, painter: QPainter, trail_idx: np.ndarray, rect: QRectF) -> None:
        if len(trail_idx) < 8:
            return
        sample = trail_idx[np.linspace(0, len(trail_idx) - 1, min(28, len(trail_idx))).astype(int)]
        pts, _ = self._project(self.data.pos_au[sample], rect)
        painter.setPen(Qt.PenStyle.NoPen)
        for n, (x, y) in enumerate(pts):
            t = n / max(len(pts) - 1, 1)
            alpha = int(18 + 120 * t)
            radius = 1.2 + 3.8 * t
            painter.setBrush(QColor(125, 249, 255, alpha))
            painter.drawEllipse(QPointF(float(x), float(y)), radius, radius)

    def _draw_anchors(self, painter: QPainter, rect: QRectF) -> None:
        if not self.data.anchors:
            return
        positions = self.data.pos_au[[anchor.nearest_index for anchor in self.data.anchors]]
        pts, _ = self._project(positions, rect)
        painter.setFont(QFont("Helvetica Neue", 9, QFont.Weight.DemiBold))
        for anchor, (x, y) in zip(self.data.anchors, pts):
            point = QPointF(float(x), float(y))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 204, 102, 36))
            painter.drawEllipse(point, 11, 11)
            painter.setPen(QPen(QColor("#050814"), 3))
            painter.setBrush(QColor("#ffcc66"))
            painter.drawEllipse(point, 6.5, 6.5)
            painter.setPen(QPen(QColor("#ffdf8b"), 1))
            painter.drawText(point + QPointF(9, -8), _short_date(anchor.label))

    def _draw_hud(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QColor(230, 242, 255, 220))
        painter.setFont(QFont("Helvetica Neue", 13, QFont.Weight.Bold))
        painter.drawText(QRectF(rect.left() + 16, rect.top() + 14, 520, 28), "Apophis Dynamics Infographic")
        painter.setFont(QFont("Helvetica Neue", 10))
        painter.setPen(QColor(190, 216, 242, 190))
        painter.drawText(
            QRectF(rect.left() + 16, rect.top() + 43, 720, 26),
            f"{_short_date(self.data.calendar[self.index])}  |  sample {self.index + 1:,}/{len(self.data.jd):,}",
        )
        if self._encounter_active():
            painter.setPen(QColor("#7df9ff"))
            painter.setFont(QFont("Helvetica Neue", 10, QFont.Weight.DemiBold))
            days = float(self.data.jd[self.index] - self.data.jd[self.data.close_index])
            painter.drawText(
                QRectF(rect.left() + 16, rect.top() + 70, 780, 24),
                f"2029 encounter frame  |  T{days:+.2f} d  |  fixed Earth-centered scale",
            )


class Sparkline(QWidget):
    def __init__(self, data: RunData) -> None:
        super().__init__()
        self.data = data
        self.metric_key = "residual"
        self.index = 0
        self.setMinimumHeight(168)

    def set_metric(self, metric: str) -> None:
        self.metric_key = metric
        self.update()

    def set_index(self, index: int) -> None:
        self.index = int(index)
        self.update()

    def _values(self) -> tuple[np.ndarray, str, QColor]:
        if self.metric_key == "integrated range":
            return self.data.integrated_dist_au * AU_KM, "Integrated geocentric distance (km)", QColor("#71d3ff")
        if self.metric_key == "horizons range":
            return self.data.horizons_dist_au * AU_KM, "Horizons geocentric distance (km)", QColor("#a8dadc")
        if self.metric_key == "gi":
            return self.data.gi_log, "log10 |GI_N|", QColor("#96f7a6")
        if self.metric_key == "oi":
            return self.data.oi_log, "log10 |OI_N|", QColor("#ffcc66")
        if self.metric_key == "speed":
            return self.data.speed_km_s, "Heliocentric speed (km/s)", QColor("#d8a2ff")
        if self.metric_key == "range delta":
            return (self.data.integrated_dist_au - self.data.horizons_dist_au) * AU_KM, "Integrated range - Horizons range (km)", QColor("#ff7b93")
        return self.data.residual_km, "Integrated minus Horizons (km)", QColor("#ff7b93")

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(14, 16, -14, -24)
        painter.fillRect(self.rect(), QColor("#07111d"))
        values, label, color = self._values()
        finite = np.isfinite(values)
        if not np.any(finite):
            return
        draw_idx = np.linspace(0, len(values) - 1, min(900, len(values))).astype(int)
        yvals = values[draw_idx]
        lo = float(np.nanpercentile(yvals, 1))
        hi = float(np.nanpercentile(yvals, 99))
        if abs(hi - lo) < 1e-12:
            hi = lo + 1.0
        x = np.linspace(rect.left(), rect.right(), len(draw_idx))
        y = rect.bottom() - (np.clip(yvals, lo, hi) - lo) / (hi - lo) * rect.height()
        path = QPainterPath(QPointF(float(x[0]), float(y[0])))
        for xx, yy in zip(x[1:], y[1:]):
            path.lineTo(float(xx), float(yy))
        painter.setPen(QPen(QColor(255, 255, 255, 28), 1))
        for frac in (0.25, 0.5, 0.75):
            yy = rect.top() + rect.height() * frac
            painter.drawLine(QPointF(rect.left(), yy), QPointF(rect.right(), yy))
        painter.setPen(QPen(color, 2.0))
        painter.drawPath(path)
        cursor_x = rect.left() + self.index / max(len(values) - 1, 1) * rect.width()
        painter.setPen(QPen(QColor("#ffffff"), 1.4))
        painter.drawLine(QPointF(cursor_x, rect.top()), QPointF(cursor_x, rect.bottom()))
        painter.setFont(QFont("Helvetica Neue", 10, QFont.Weight.DemiBold))
        painter.setPen(QColor("#eef6ff"))
        painter.drawText(QRectF(rect.left(), 2, rect.width(), 18), label)
        painter.setFont(QFont("Helvetica Neue", 9))
        painter.setPen(QColor(200, 216, 232, 180))
        painter.drawText(QRectF(rect.left(), rect.bottom() + 4, rect.width(), 18), f"p01={lo:,.3g}   p99={hi:,.3g}")


class CloseApproachDashboard(QWidget):
    def __init__(self, data: RunData) -> None:
        super().__init__()
        self.data = data
        self.index = int(data.close_index)
        self.lunar_distance_km = 384_400.0
        self.earth_radius_km = 6_371.0
        self.days = data.jd - data.jd[data.close_index]
        self.window = np.abs(self.days) <= 35.0
        if int(np.count_nonzero(self.window)) < 8:
            self.window = np.ones_like(self.days, dtype=bool)
        self.geom_window = np.abs(self.days) <= 1.0
        if int(np.count_nonzero(self.geom_window)) < 4:
            self.geom_window = self.window
        self.range_rate = data.range_rate_km_s
        self.geo_re = (data.pos_au - data.earth_fixed_au) * AU_KM / self.earth_radius_km
        self.setMinimumHeight(400)

    def set_index(self, index: int) -> None:
        self.index = int(max(0, min(len(self.data.jd) - 1, index)))
        self.update()

    def _panel_rects(self) -> list[QRectF]:
        outer = QRectF(self.rect()).adjusted(58, 10, -18, -30)
        gap = 42.0
        title_h = 28.0
        w = (outer.width() - gap) / 2.0
        h = (outer.height() - title_h - gap) / 2.0
        top = outer.top() + title_h
        return [
            QRectF(outer.left(), top, w, h),
            QRectF(outer.left() + w + gap, top, w, h),
            QRectF(outer.left(), top + h + gap, w, h),
            QRectF(outer.left() + w + gap, top + h + gap, w, h),
        ]

    def _map_x(self, rect: QRectF, days: np.ndarray | float) -> np.ndarray | float:
        lo, hi = -35.0, 35.0
        return rect.left() + (np.asarray(days) - lo) / (hi - lo) * rect.width()

    def _map_y_linear(self, rect: QRectF, values: np.ndarray | float, lo: float, hi: float) -> np.ndarray | float:
        hi = hi if abs(hi - lo) > 1e-12 else lo + 1.0
        return rect.bottom() - (np.asarray(values) - lo) / (hi - lo) * rect.height()

    def _map_y_log(self, rect: QRectF, values: np.ndarray | float, lo: float, hi: float) -> np.ndarray | float:
        vlo = math.log10(max(lo, 1e-6))
        vhi = math.log10(max(hi, lo * 10.0))
        arr = np.log10(np.maximum(np.asarray(values), 1e-6))
        return rect.bottom() - (arr - vlo) / max(vhi - vlo, 1e-12) * rect.height()

    def _draw_frame(self, painter: QPainter, rect: QRectF, title: str, label: str) -> None:
        painter.fillRect(rect, QColor("#fbfbfb"))
        painter.setPen(QPen(QColor("#d7d7d7"), 1))
        for frac in (0.25, 0.5, 0.75):
            x = rect.left() + rect.width() * frac
            y = rect.top() + rect.height() * frac
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        painter.setPen(QPen(QColor("#222222"), 1.2))
        painter.drawRect(rect)
        painter.setFont(QFont("Helvetica Neue", 12, QFont.Weight.DemiBold))
        painter.drawText(QRectF(rect.left(), rect.top() - 22, rect.width(), 20), Qt.AlignmentFlag.AlignCenter, title)
        badge = QRectF(rect.left() + 8, rect.top() + 8, 22, 22)
        painter.setBrush(QColor("#f7f7f7"))
        painter.setPen(QPen(QColor("#cfcfcf"), 1))
        painter.drawRoundedRect(badge, 4, 4)
        painter.setPen(QColor("#111111"))
        painter.setFont(QFont("Helvetica Neue", 10, QFont.Weight.Bold))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, label)

    def _draw_time_cursor(self, painter: QPainter, rect: QRectF, color: QColor = QColor("#d1495b")) -> float | None:
        day = float(self.days[self.index])
        if day < -35.0 or day > 35.0:
            return None
        cursor_x = float(self._map_x(rect, day))
        painter.setPen(QPen(color, 1.1))
        painter.drawLine(QPointF(cursor_x, rect.top()), QPointF(cursor_x, rect.bottom()))
        return cursor_x

    def _draw_axis_text(
        self,
        painter: QPainter,
        rect: QRectF,
        x_label: str,
        y_label: str,
        y_min: float,
        y_max: float,
        show_limits: bool = True,
    ) -> None:
        painter.setFont(QFont("Helvetica Neue", 8))
        painter.setPen(QColor("#333333"))
        painter.drawText(QRectF(rect.left(), rect.bottom() + 2, rect.width(), 14), Qt.AlignmentFlag.AlignCenter, x_label)
        painter.save()
        painter.translate(rect.left() - 30, rect.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-rect.height() / 2, -7, rect.height(), 14), Qt.AlignmentFlag.AlignCenter, y_label)
        painter.restore()
        if show_limits:
            painter.setPen(QColor("#555555"))
            painter.drawText(QRectF(rect.left() - 44, rect.top() - 2, 40, 12), Qt.AlignmentFlag.AlignRight, f"{y_max:.3g}")
            painter.drawText(QRectF(rect.left() - 44, rect.bottom() - 10, 40, 12), Qt.AlignmentFlag.AlignRight, f"{y_min:.3g}")

    def _nice_limit(self, value: float) -> float:
        value = abs(float(value))
        if not math.isfinite(value) or value <= 0.0:
            return 1.0
        exponent = math.floor(math.log10(value))
        fraction = value / (10.0**exponent)
        if fraction <= 1.5:
            nice = 1.5
        elif fraction <= 2.0:
            nice = 2.0
        elif fraction <= 3.0:
            nice = 3.0
        elif fraction <= 5.0:
            nice = 5.0
        else:
            nice = 10.0
        return nice * (10.0**exponent)

    def _draw_y_ticks(self, painter: QPainter, rect: QRectF, ticks: list[float], lo: float, hi: float) -> None:
        painter.setFont(QFont("Helvetica Neue", 8))
        for tick in ticks:
            y = float(self._map_y_linear(rect, tick, lo, hi))
            painter.setPen(QPen(QColor("#dedede"), 1))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            painter.setPen(QColor("#555555"))
            painter.drawText(QRectF(rect.left() - 46, y - 7, 42, 14), Qt.AlignmentFlag.AlignRight, f"{tick:g}")

    def _draw_value_badge(self, painter: QPainter, rect: QRectF, text: str, color: QColor) -> None:
        painter.setFont(QFont("Helvetica Neue", 9, QFont.Weight.DemiBold))
        badge = QRectF(rect.right() - 154, rect.top() + 10, 144, 24)
        painter.setPen(QPen(color, 1))
        fill = QColor(color)
        fill.setAlpha(34)
        painter.setBrush(fill)
        painter.drawRoundedRect(badge, 5, 5)
        painter.setPen(color.darker(130))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_segmented_path(
        self,
        painter: QPainter,
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray,
        positive_color: QColor,
        negative_color: QColor,
        width: float,
    ) -> None:
        if len(x) < 2:
            return
        start = 0
        current_positive = values[0] >= 0.0
        for i in range(1, len(x)):
            is_positive = values[i] >= 0.0
            if is_positive != current_positive:
                color = positive_color if current_positive else negative_color
                self._draw_path(painter, x[start : i + 1], y[start : i + 1], color, width)
                start = i
                current_positive = is_positive
        color = positive_color if current_positive else negative_color
        self._draw_path(painter, x[start:], y[start:], color, width)

    def _draw_path(self, painter: QPainter, x: np.ndarray, y: np.ndarray, color: QColor, width: float, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> None:
        if len(x) < 2:
            return
        path = QPainterPath(QPointF(float(x[0]), float(y[0])))
        for xx, yy in zip(x[1:], y[1:]):
            path.lineTo(float(xx), float(yy))
        pen = QPen(color, width)
        pen.setStyle(style)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawPath(path)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#f5f6f8"))
        outer = QRectF(self.rect()).adjusted(12, 2, -12, -2)
        painter.setPen(QColor("#101010"))
        painter.setFont(QFont("Helvetica Neue", 15, QFont.Weight.DemiBold))
        painter.drawText(QRectF(outer.left(), outer.top(), outer.width(), 24), Qt.AlignmentFlag.AlignCenter, "Live Close-Approach Dashboard")
        rect_a, rect_b, rect_c, rect_d = self._panel_rects()
        self._draw_range_panel(painter, rect_a)
        self._draw_rate_panel(painter, rect_b)
        self._draw_geometry_panel(painter, rect_c)
        self._draw_residual_panel(painter, rect_d)

    def _draw_range_panel(self, painter: QPainter, rect: QRectF) -> None:
        self._draw_frame(painter, rect, "How Close Is the Encounter?", "A")
        days = self.days[self.window]
        integrated_ld = self.data.integrated_dist_au[self.window] * AU_KM / self.lunar_distance_km
        horizons_ld = self.data.horizons_dist_au[self.window] * AU_KM / self.lunar_distance_km
        hi = max(float(np.nanpercentile(np.r_[integrated_ld, horizons_ld], 99.5)) * 1.12, 1.2)
        lo = max(min(float(np.nanmin(integrated_ld)), float(np.nanmin(horizons_ld)), 0.08), 0.05)
        x = self._map_x(rect, days)
        self._draw_path(painter, x, self._map_y_log(rect, integrated_ld, lo, hi), QColor("#29495a"), 2.0)
        self._draw_path(painter, x, self._map_y_log(rect, horizons_ld, lo, hi), QColor("#2a9d8f"), 1.6, Qt.PenStyle.DashLine)
        moon_y = float(self._map_y_log(rect, 1.0, lo, hi))
        painter.setPen(QPen(QColor("#888888"), 1.0, Qt.PenStyle.DotLine))
        painter.drawLine(QPointF(rect.left(), moon_y), QPointF(rect.right(), moon_y))
        self._draw_axis_text(painter, rect, "Days from nearest CAD epoch", "Lunar distances", lo, hi)
        cursor_x = self._draw_time_cursor(painter, rect)
        if cursor_x is not None:
            current_ld = float(self.data.integrated_dist_au[self.index] * AU_KM / self.lunar_distance_km)
            painter.setBrush(QColor("#d1495b"))
            painter.drawEllipse(QPointF(cursor_x, float(self._map_y_log(rect, current_ld, lo, hi))), 4.5, 4.5)

    def _draw_rate_panel(self, painter: QPainter, rect: QRectF) -> None:
        self._draw_frame(painter, rect, "Inbound / Outbound Range Rate", "B")
        days = self.days[self.window]
        values = self.range_rate[self.window]
        finite = values[np.isfinite(values)]
        limit = self._nice_limit(max(float(np.nanmax(np.abs(finite))) * 1.05, 1.0)) if len(finite) else 1.0
        x = self._map_x(rect, days)
        y = self._map_y_linear(rect, values, -limit, limit)
        zero_y = float(self._map_y_linear(rect, 0.0, -limit, limit))

        painter.fillRect(QRectF(rect.left(), rect.top(), rect.width(), max(0.0, zero_y - rect.top())), QColor("#fff2f4"))
        painter.fillRect(QRectF(rect.left(), zero_y, rect.width(), max(0.0, rect.bottom() - zero_y)), QColor("#eefaf8"))
        self._draw_y_ticks(painter, rect, [-limit, -limit / 2.0, 0.0, limit / 2.0, limit], -limit, limit)
        painter.setPen(QPen(QColor("#6a6a6a"), 1.5))
        painter.drawLine(QPointF(rect.left(), zero_y), QPointF(rect.right(), zero_y))
        painter.setFont(QFont("Helvetica Neue", 8, QFont.Weight.DemiBold))
        painter.setPen(QColor(161, 45, 65, 150))
        painter.drawText(QRectF(rect.left() + 8, rect.top() + 34, 100, 16), Qt.AlignmentFlag.AlignLeft, "OUTBOUND")
        painter.setPen(QColor(24, 117, 108, 150))
        painter.drawText(QRectF(rect.left() + 8, rect.bottom() - 22, 100, 16), Qt.AlignmentFlag.AlignLeft, "INBOUND")
        self._draw_segmented_path(painter, x, y, values, QColor("#d1495b"), QColor("#2a9d8f"), 2.6)
        self._draw_axis_text(painter, rect, "Days from nearest CAD epoch", "km/s", -limit, limit, False)
        cursor_x = self._draw_time_cursor(painter, rect, QColor("#111111"))
        if cursor_x is not None:
            current_rate = float(self.range_rate[self.index])
            color = QColor("#d1495b") if current_rate >= 0.0 else QColor("#2a9d8f")
            cy = float(self._map_y_linear(rect, np.clip(current_rate, -limit, limit), -limit, limit))
            painter.setPen(QPen(QColor("#111111"), 1.0))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(cursor_x, cy), 5.0, 5.0)
            direction = "outbound" if current_rate >= 0.0 else "inbound"
            self._draw_value_badge(painter, rect, f"{direction} {current_rate:+.2f}", color)

    def _draw_geometry_panel(self, painter: QPainter, rect: QRectF) -> None:
        self._draw_frame(painter, rect, "Earth-Moon Scale Encounter Geometry", "C")
        geom = self.geo_re[self.geom_window, :2]
        lim = max(float(np.nanmax(np.abs(geom))) * 1.08, self.lunar_distance_km / self.earth_radius_km * 1.08, 10.0)
        x = rect.center().x() + geom[:, 0] / lim * rect.width() * 0.46
        y = rect.center().y() - geom[:, 1] / lim * rect.height() * 0.46
        self._draw_path(painter, x, y, QColor("#29495a"), 2.0)
        painter.setBrush(QColor(93, 178, 242, 120))
        painter.setPen(QPen(QColor("#2a9df4"), 1.0))
        painter.drawEllipse(rect.center(), 5.5, 5.5)
        moon_r = self.lunar_distance_km / self.earth_radius_km / lim * rect.width() * 0.46
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#8a8a8a"), 1.2, Qt.PenStyle.DashLine))
        painter.drawEllipse(rect.center(), moon_r, moon_r)
        self._draw_axis_text(painter, rect, "Geocentric X (Earth radii)", "Geocentric Y", -lim, lim)
        current = (self.data.pos_au[self.index] - self.data.earth_fixed_au) * AU_KM / self.earth_radius_km
        cx = rect.center().x() + float(current[0]) / lim * rect.width() * 0.46
        cy = rect.center().y() - float(current[1]) / lim * rect.height() * 0.46
        painter.setBrush(QColor("#d1495b"))
        painter.setPen(QPen(QColor("#111111"), 1.2))
        painter.drawEllipse(QPointF(cx, cy), 6.5, 6.5)

    def _draw_residual_panel(self, painter: QPainter, rect: QRectF) -> None:
        self._draw_frame(painter, rect, "Prediction Residual, Focused", "D")
        days = self.days[self.window]
        values = self.data.residual_km[self.window]
        focus = self.window & (np.abs(self.days) <= 12.0)
        scale_values = self.data.residual_km[focus]
        scale_values = scale_values[np.isfinite(scale_values)]
        if len(scale_values):
            p2, p98 = np.nanpercentile(scale_values, [2.0, 98.0])
            spread = max(abs(float(p2)), abs(float(p98)), 100.0)
        else:
            finite = values[np.isfinite(values)]
            spread = max(float(np.nanmax(np.abs(finite))) if len(finite) else 100.0, 100.0)
        current_day = float(self.days[self.index])
        current_residual = float(self.data.residual_km[self.index])
        if abs(current_day) <= 12.0 and math.isfinite(current_residual):
            spread = max(spread, abs(current_residual))
        limit = self._nice_limit(spread * 1.15)
        lo, hi = -limit, limit
        clipped_values = np.clip(values, lo, hi)
        x = self._map_x(rect, days)
        y = self._map_y_linear(rect, clipped_values, lo, hi)
        zero_y = float(self._map_y_linear(rect, 0.0, lo, hi))

        painter.fillRect(QRectF(rect.left(), rect.top(), rect.width(), max(0.0, zero_y - rect.top())), QColor("#f4fbef"))
        painter.fillRect(QRectF(rect.left(), zero_y, rect.width(), max(0.0, rect.bottom() - zero_y)), QColor("#f0f6fb"))
        self._draw_y_ticks(painter, rect, [-limit, -limit / 2.0, 0.0, limit / 2.0, limit], lo, hi)
        painter.setPen(QPen(QColor("#6a6a6a"), 1.4))
        painter.drawLine(QPointF(rect.left(), zero_y), QPointF(rect.right(), zero_y))
        self._draw_segmented_path(painter, x, y, clipped_values, QColor("#6a994e"), QColor("#457b9d"), 2.5)
        clipped_high = values > hi
        clipped_low = values < lo
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(209, 73, 91, 130))
        for xx in x[clipped_high]:
            painter.drawEllipse(QPointF(float(xx), rect.top() + 4.0), 2.2, 2.2)
        painter.setBrush(QColor(69, 123, 157, 130))
        for xx in x[clipped_low]:
            painter.drawEllipse(QPointF(float(xx), rect.bottom() - 4.0), 2.2, 2.2)
        painter.setFont(QFont("Helvetica Neue", 8, QFont.Weight.DemiBold))
        painter.setPen(QColor(80, 80, 80, 150))
        painter.drawText(QRectF(rect.left() + 8, rect.top() + 34, rect.width() - 16, 16), Qt.AlignmentFlag.AlignLeft, "focused scale; dots mark clipped tails")
        self._draw_axis_text(painter, rect, "Days from nearest CAD epoch", "km", lo, hi, False)
        cursor_x = self._draw_time_cursor(painter, rect)
        if cursor_x is not None:
            plot_residual = float(np.clip(current_residual, lo, hi))
            color = QColor("#6a994e") if current_residual >= 0.0 else QColor("#457b9d")
            painter.setPen(QPen(QColor("#111111"), 1.0))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(cursor_x, float(self._map_y_linear(rect, plot_residual, lo, hi))), 5.0, 5.0)
            suffix = " clipped" if current_residual < lo or current_residual > hi else ""
            self._draw_value_badge(painter, rect, f"{current_residual:+,.0f} km{suffix}", color)


class DetailPanel(QWidget):
    def __init__(self, data: RunData) -> None:
        super().__init__()
        self.data = data
        dyn = data.report.get("dynamics", {})
        nd = dyn.get("numerical_diagnostics", {})
        cadence = str(dyn.get("horizons_step", "n/a"))
        prediction_mode = str(nd.get("prediction_mode", "n/a"))
        refresh_count = int(float(nd.get("state_refresh_count", 0.0))) if nd.get("state_refresh_count") is not None else 0
        refresh_days = float(nd.get("state_refresh_segment_days", 0.0) or 0.0)
        refresh_text = "none" if refresh_count <= 0 else f"{refresh_count} / {refresh_days:g} d"
        cad_anchor_rmse = nd.get("cad_anchor_integrated_rmse_km", dyn.get("validation_rmse_km", float("nan")))
        self.trust_cards: dict[str, StatCard] = {
            "mode": StatCard("Prediction Mode", prediction_mode.replace("_", " "), "#ffd166"),
            "refresh": StatCard("State Refresh", refresh_text, "#ffb703"),
            "val_rmse": StatCard("Validation RMSE", _fmt_km(float(dyn.get("validation_rmse_km", float("nan")))), "#a8dadc"),
            "cad_error": StatCard("Nearest CAD Error", _fmt_km(float(dyn.get("cad_validation_error_km", float("nan")))), "#7df9ff"),
            "cad_rmse": StatCard("CAD Anchor RMSE", _fmt_km(float(cad_anchor_rmse)), "#b8f2e6"),
            "cadence": StatCard("Sample Cadence", cadence, "#cdb4db"),
            "global_gate": StatCard("Global ML Gate", _fmt_gate(nd.get("global_residual_gate_accepted")), "#f4a261"),
            "local_gate": StatCard("Local ML Gate", _fmt_gate(nd.get("local_gate_accepted")), "#e9c46a"),
            "tf_gate": StatCard("CAD Recon Gate", _fmt_gate(nd.get("tensorflow_continuous_anchor_gate_accepted")), "#d8a2ff"),
            "samples": StatCard("Samples", f"{int(dyn.get('n_samples', len(data.jd))):,}", "#96f7a6"),
        }
        self.cards: dict[str, StatCard] = {
            "date": StatCard("Current Epoch", "n/a", "#71d3ff"),
            "distance": StatCard("Integrated Range", "n/a", "#96f7a6"),
            "horizons": StatCard("Horizons Range", "n/a", "#a8dadc"),
            "residual": StatCard("Horizons Residual", "n/a", "#ff7b93"),
            "speed": StatCard("Heliocentric Speed", "n/a", "#d8a2ff"),
            "gi": StatCard("log10 |GI_N|", "n/a", "#ffcc66"),
            "oi": StatCard("log10 |OI_N|", "n/a", "#ff9f7a"),
            "cov": StatCard("Median Cov90", _fmt_km(float(nd.get("covariance_width90_km_median", float("nan")))), "#b8f2e6"),
            "cascade": StatCard("Cascade Accel", f"{float(nd.get('cascade_acceleration_au_d2_median', float('nan'))):.3e} au/d2", "#f4a261"),
            "phase": StatCard("Phase Accel", f"{float(nd.get('phase_acceleration_au_d2_median', float('nan'))):.3e} au/d2", "#e9c46a"),
            "solar_gr": StatCard("Solar GR Accel", f"{float(nd.get('solar_gr_acceleration_au_d2_median', float('nan'))):.3e} au/d2", "#e76f51"),
            "nongrav": StatCard("A1/A2 Accel", f"{float(nd.get('standard_nongrav_acceleration_au_d2_median', float('nan'))):.3e} au/d2", "#2a9d8f"),
            "frame": StatCard("Dynamics Frame", str(nd.get("dynamics_frame", "n/a")), "#bde0fe"),
        }
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        trust_title = QLabel("Trust / Validation")
        trust_title.setObjectName("panelTitle")
        layout.addWidget(trust_title)
        trust_grid = QGridLayout()
        trust_grid.setHorizontalSpacing(10)
        trust_grid.setVerticalSpacing(10)
        for i, card in enumerate(self.trust_cards.values()):
            trust_grid.addWidget(card, i // 2, i % 2)
        layout.addLayout(trust_grid)

        title = QLabel("Live Dynamics Fields")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for i, card in enumerate(self.cards.values()):
            grid.addWidget(card, i // 2, i % 2)
        layout.addLayout(grid)
        self.anchor_label = QLabel()
        self.anchor_label.setObjectName("anchorBox")
        self.anchor_label.setWordWrap(True)
        layout.addWidget(self.anchor_label)
        layout.addStretch(1)
        self.setMinimumWidth(370)

    def update_index(self, index: int) -> None:
        data = self.data
        self.cards["date"].set_value(_short_date(data.calendar[index]))
        self.cards["distance"].set_value(_fmt_km(float(data.integrated_dist_au[index] * AU_KM)))
        self.cards["horizons"].set_value(_fmt_km(float(data.horizons_dist_au[index] * AU_KM)))
        self.cards["residual"].set_value(_fmt_km(float(data.residual_km[index])))
        self.cards["speed"].set_value(f"{float(data.speed_km_s[index]):.3f} km/s")
        self.cards["gi"].set_value(f"{float(data.gi_log[index]):.3f}")
        self.cards["oi"].set_value(f"{float(data.oi_log[index]):.3f}")
        if data.anchors:
            nearest = min(data.anchors, key=lambda anchor: abs(anchor.nearest_index - index))
            self.anchor_label.setText(
                "Nearest CAD anchor\n"
                f"{nearest.label}\n"
                f"CAD range: {_fmt_km(nearest.cad_distance_km)}\n"
                f"Integrated - CAD: {_fmt_km(nearest.integrated_minus_cad_km)}\n"
                f"Horizons - CAD: {_fmt_km(nearest.horizons_minus_cad_km)}"
            )


class MainWindow(QMainWindow):
    def __init__(self, data: RunData) -> None:
        super().__init__()
        self.data = data
        self.setWindowTitle("ASTEROID-NEO Immersive Dynamics Infographic")
        self.resize(1480, 940)
        self.canvas = TrajectoryCanvas(data)
        self.details = DetailPanel(data)
        self.sparkline = Sparkline(data)
        self.dashboard = CloseApproachDashboard(data)
        self.play_button = QPushButton("Pause")
        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.trail_slider = QSlider(Qt.Orientation.Horizontal)
        self.metric_combo = QComboBox()
        self.view_combo = QComboBox()
        self.auto_rotate = QCheckBox("Auto rotate")
        self._build_ui()
        self._connect()
        self.canvas.play()
        self.details.update_index(0)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = self._make_header()
        root.addWidget(header)
        controls = self._make_controls()
        root.addWidget(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self.canvas, 2)
        left_layout.addWidget(self.dashboard, 1)
        splitter.addWidget(left)
        detail_scroll = QScrollArea()
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setFrameShape(QFrame.Shape.NoFrame)
        detail_scroll.setMinimumHeight(0)
        detail_scroll.setMinimumWidth(430)
        detail_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored)
        detail_scroll.setWidget(self.details)
        splitter.addWidget(detail_scroll)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self._make_menu()
        self._apply_style()

    def _make_header(self) -> QWidget:
        dyn = self.data.report.get("dynamics", {})
        obj = self.data.report.get("object", {})
        header = QFrame()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 12, 18, 12)
        text = QLabel(
            f"{obj.get('fullname', '99942 Apophis')}  |  "
            f"{dyn.get('n_samples', len(self.data.jd)):,} samples  |  "
            f"Nearest CAD error {_fmt_km(float(dyn.get('cad_validation_error_km', float('nan'))))}"
        )
        text.setObjectName("headerTitle")
        bundle = QLabel(str(self.data.bundle))
        bundle.setObjectName("headerPath")
        layout.addWidget(text, 1)
        layout.addWidget(bundle)
        return header

    def _make_controls(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("controls")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)
        self.scrubber.setRange(0, len(self.data.jd) - 1)
        self.scrubber.setSingleStep(max(1, len(self.data.jd) // 1000))
        self.speed_slider.setRange(1, max(2, len(self.data.jd) // 90))
        self.speed_slider.setValue(self.canvas.step)
        self.trail_slider.setRange(80, min(3000, len(self.data.jd)))
        self.trail_slider.setValue(self.canvas.trail_points)
        self.auto_rotate.setChecked(False)
        self.view_combo.addItems(["Solar frame", "2029 encounter"])
        self.metric_combo.addItems(["residual", "range delta", "integrated range", "horizons range", "speed", "gi", "oi"])

        layout.addWidget(self.play_button)
        layout.addWidget(QLabel("View"))
        layout.addWidget(self.view_combo)
        layout.addWidget(QLabel("Scrub"))
        layout.addWidget(self.scrubber, 4)
        layout.addWidget(QLabel("Speed"))
        layout.addWidget(self.speed_slider, 1)
        layout.addWidget(QLabel("Trail"))
        layout.addWidget(self.trail_slider, 1)
        layout.addWidget(QLabel("Field"))
        layout.addWidget(self.metric_combo)
        layout.addWidget(self.auto_rotate)
        return panel

    def _make_menu(self) -> None:
        toolbar = QToolBar("Run")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        open_action = QAction("Open Bundle", self)
        open_action.triggered.connect(self._open_bundle)
        toolbar.addAction(open_action)
        reset_action = QAction("Reset Camera", self)
        reset_action.triggered.connect(self._reset_camera)
        toolbar.addAction(reset_action)
        jump_action = QAction("Jump 2029 Encounter", self)
        jump_action.triggered.connect(self._jump_close_approach)
        toolbar.addAction(jump_action)

    def _connect(self) -> None:
        self.canvas.indexChanged.connect(self.scrubber.setValue)
        self.canvas.indexChanged.connect(self.details.update_index)
        self.canvas.indexChanged.connect(self.sparkline.set_index)
        self.canvas.indexChanged.connect(self.dashboard.set_index)
        self.scrubber.valueChanged.connect(self.canvas.set_index)
        self.speed_slider.valueChanged.connect(self.canvas.set_speed)
        self.trail_slider.valueChanged.connect(self.canvas.set_trail)
        self.auto_rotate.toggled.connect(self.canvas.set_auto_rotate)
        self.view_combo.currentTextChanged.connect(self.canvas.set_view_mode)
        self.metric_combo.currentTextChanged.connect(self.sparkline.set_metric)
        self.play_button.clicked.connect(self._toggle_play)

    def _toggle_play(self) -> None:
        running = not self.canvas.timer.isActive()
        self.canvas.set_running(running)
        self.play_button.setText("Pause" if running else "Play")

    def _reset_camera(self) -> None:
        self.canvas.yaw = -0.72
        self.canvas.pitch = 0.52
        self.canvas.zoom = 0.92
        self.canvas.update()

    def _jump_close_approach(self) -> None:
        self.canvas.pause()
        self.play_button.setText("Play")
        self.view_combo.setCurrentText("2029 encounter")
        self.canvas.set_index(self.data.close_index)
        self.canvas.zoom = 1.25
        self.canvas.update()

    def _open_bundle(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open ASTEROID-NEO output bundle", str(self.data.bundle))
        if not path:
            return
        try:
            data = load_run(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Could not open bundle", str(exc))
            return
        self.canvas.pause()
        new_window = MainWindow(data)
        new_window.show()
        self.close()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #07111d;
                color: #eaf4ff;
                font-family: "Helvetica Neue", Arial, sans-serif;
            }
            QFrame#header {
                background: #0a1624;
                border-bottom: 1px solid rgba(255,255,255,38);
            }
            QLabel#headerTitle {
                font-size: 18px;
                font-weight: 800;
                color: #f7fbff;
            }
            QLabel#headerPath {
                color: rgba(216,229,242,165);
                font-size: 11px;
            }
            QLabel#panelTitle {
                font-size: 18px;
                font-weight: 800;
                color: #f7fbff;
                padding-bottom: 6px;
            }
            QLabel#anchorBox {
                border: 1px solid rgba(255,255,255,30);
                background: rgba(255,255,255,9);
                border-radius: 8px;
                padding: 12px;
                color: #dcecff;
                font-size: 13px;
                line-height: 1.35;
            }
            QFrame#controls {
                background: #0a1624;
                border-top: 1px solid rgba(255,255,255,32);
            }
            QPushButton {
                background: #163557;
                border: 1px solid #2b6da6;
                border-radius: 7px;
                padding: 8px 16px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #1e4771;
            }
            QSlider::groove:horizontal {
                height: 5px;
                background: #26384e;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #71d3ff;
                width: 15px;
                margin: -6px 0;
                border-radius: 7px;
            }
            QComboBox {
                background: #101f31;
                border: 1px solid #33475f;
                padding: 6px 10px;
                border-radius: 6px;
            }
            QCheckBox {
                spacing: 8px;
            }
            """
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animated PyQt6 infographic viewer for ASTEROID-NEO run outputs.")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
        help="Output directory containing report.json and the generated dynamics CSV tables.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    data = load_run(args.bundle)
    app = QApplication(sys.argv[:1])
    app.setStyle(QStyleFactory.create("Fusion"))
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#07111d"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#eaf4ff"))
    app.setPalette(palette)
    window = MainWindow(data)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
