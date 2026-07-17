"""Streamlit Multipage Entry Point — Detection."""

from __future__ import annotations

import streamlit as st

from src.dashboard.config import load_dashboard_settings
from src.dashboard.detection_page import render_detection_page
from src.dashboard.styles import inject_dashboard_styles


st.set_page_config(
    page_title="Manufacturing Vision Detection",
    page_icon="🎯",
    layout="wide",
)

settings = load_dashboard_settings()
inject_dashboard_styles(st)
render_detection_page(settings=settings)
