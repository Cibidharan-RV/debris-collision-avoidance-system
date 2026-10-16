"""Interactive 3D orbit visualization using Plotly."""

import os
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from src.constants import R_EARTH, SAFETY_DISTANCE_KM

class OrbitVisualizer:
    """Builds 3D orbit plots with Plotly."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _create_earth_mesh(self, resolution: int = 40) -> go.Surface:
        """Returns a 3D sphere surface representing Earth."""
        phi = np.linspace(0, np.pi, resolution)
        theta = np.linspace(0, 2 * np.pi, resolution)
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        
        x = R_EARTH * np.sin(phi_grid) * np.cos(theta_grid)
        y = R_EARTH * np.sin(phi_grid) * np.sin(theta_grid)
        z = R_EARTH * np.cos(phi_grid)
        
        colors = np.sin(theta_grid * 3.0) * np.cos(phi_grid * 2.0)
        
        return go.Surface(
            x=x, y=y, z=z,
            surfacecolor=colors,
            colorscale=[
                [0.0, "#0b1d3a"],
                [0.4, "#153966"],
                [0.6, "#225e52"],
                [0.8, "#524430"],
                [1.0, "#d0e0eb"]
            ],
            showscale=False,
            opacity=0.95,
            name="Earth (R=6378 km)",
            hoverinfo="name"
        )

    def generate_3d_encounter_scene(
        self,
        event,
        sat_traj: np.ndarray,
        deb_traj: np.ndarray,
        post_burn_sat_traj: np.ndarray | None = None,
        burn_point: np.ndarray | None = None,
        delta_v_eci: np.ndarray | None = None,
        filename: str = "encounter_3d_visualization.html"
    ) -> str:
        """Generates interactive 3D HTML plot of encounter and avoidance maneuver."""
        fig = go.Figure()

        fig.add_trace(self._create_earth_mesh())

        fig.add_trace(go.Scatter3d(
            x=sat_traj[0, :],
            y=sat_traj[1, :],
            z=sat_traj[2, :],
            mode='lines',
            line=dict(color='#00d2ff', width=5),
            name=f"Primary: {event.primary_name} (Pre-burn)",
            hoverinfo='name'
        ))

        fig.add_trace(go.Scatter3d(
            x=deb_traj[0, :],
            y=deb_traj[1, :],
            z=deb_traj[2, :],
            mode='lines',
            line=dict(color='#ff2a6d', width=5, dash='dash'),
            name=f"Debris: {event.secondary_name}",
            hoverinfo='name'
        ))

        r_tca = event.r_primary_tca
        fig.add_trace(go.Scatter3d(
            x=[r_tca[0]],
            y=[r_tca[1]],
            z=[r_tca[2]],
            mode='markers+text',
            marker=dict(size=9, color='#ff0055', symbol='diamond'),
            text=[f"TCA Hazard ({event.miss_distance_km:.2f} km)"],
            textposition="top center",
            name="TCA Conjunction Point",
            hovertext=f"Miss Distance: {event.miss_distance_km:.4f} km<br>TCA: {event.tca_epoch.strftime('%H:%M:%S UTC')}"
        ))

        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        display_scale = 150.0
        xs = r_tca[0] + display_scale * np.outer(np.cos(u), np.sin(v))
        ys = r_tca[1] + display_scale * np.outer(np.sin(u), np.sin(v))
        zs = r_tca[2] + display_scale * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=xs, y=ys, z=zs,
            colorscale=[[0, 'rgba(255, 60, 0, 0.25)'], [1, 'rgba(255, 60, 0, 0.25)']],
            showscale=False,
            opacity=0.35,
            name=f"Safety Indicator [{event.risk_threshold_km:.1f} km threshold, magnified 150x for display]",
            hoverinfo="name+text",
            hovertext=f"Physical Safety Threshold: {event.risk_threshold_km:.1f} km (magnified 150x to {display_scale:.0f} km radius for visibility against Earth)"
        ))

        if post_burn_sat_traj is not None:
            fig.add_trace(go.Scatter3d(
                x=post_burn_sat_traj[0, :],
                y=post_burn_sat_traj[1, :],
                z=post_burn_sat_traj[2, :],
                mode='lines',
                line=dict(color='#05ffa1', width=6),
                name="Post-Burn Safe Trajectory (Verified under model)",
                hoverinfo='name'
            ))

        if burn_point is not None and delta_v_eci is not None:
            dv_scale = 200000.0
            dv_end = burn_point + delta_v_eci * dv_scale
            dv_mag_m_s = np.linalg.norm(delta_v_eci) * 1000.0
            fig.add_trace(go.Scatter3d(
                x=[burn_point[0], dv_end[0]],
                y=[burn_point[1], dv_end[1]],
                z=[burn_point[2], dv_end[2]],
                mode='lines+markers',
                line=dict(color='#ffe600', width=7),
                marker=dict(size=[8, 4], color='#ffe600'),
                name="ΔV Thrust Vector [Direction indicator, scaled]",
                hovertext=f"Impulse Magnitude: {dv_mag_m_s:.2f} m/s (GCRS inertial direction)"
            ))

        fig.update_layout(
            title=dict(
                text=(
                    f"<b>LEO Debris Collision Avoidance System — 3D Encounter Geometry</b><br>"
                    f"<span style='font-size:13px; color:#a0aec0;'>"
                    f"Primary: {event.primary_name} | Secondary: {event.secondary_name} | "
                    f"Predicted Miss Distance: {event.miss_distance_km:.4f} km | Frame: GCRS (J2000 Inertial)</span>"
                ),
                font=dict(family="Segoe UI, Inter, sans-serif", size=18, color="#ffffff"),
                x=0.03, y=0.95
            ),
            paper_bgcolor="#080c14",
            plot_bgcolor="#080c14",
            scene=dict(
                xaxis=dict(title="X [km] (GCRS / J2000 Inertial)", backgroundcolor="#080c14", gridcolor="#1a2333", showbackground=True),
                yaxis=dict(title="Y [km] (GCRS / J2000 Inertial)", backgroundcolor="#080c14", gridcolor="#1a2333", showbackground=True),
                zaxis=dict(title="Z [km] (GCRS / J2000 Inertial)", backgroundcolor="#080c14", gridcolor="#1a2333", showbackground=True),
                aspectmode="data",
                camera=dict(
                    eye=dict(x=1.35, y=1.35, z=0.95)
                )
            ),
            legend=dict(
                font=dict(color="#ffffff", size=11),
                bgcolor="rgba(15, 23, 42, 0.75)",
                bordercolor="#334155",
                borderwidth=1,
                x=0.02, y=0.1
            ),
            annotations=[
                dict(
                    text="Scale Note: Safety sphere (150x) and ΔV vector are magnified for visual clarity against Earth radius (6,378 km). All numerical distance labels are true unmagnified physical distances in km.",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.03, y=0.02,
                    font=dict(size=10, color="#94a3b8")
                )
            ],
            margin=dict(l=0, r=0, b=0, t=60)
        )

        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path, include_plotlyjs="cdn")
        return output_path
