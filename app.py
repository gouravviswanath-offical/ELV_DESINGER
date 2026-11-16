import streamlit as st
import pandas as pd
import numpy as np
import cv2
import random
import json
import tempfile
import os

from utils.calculations import *
from utils.file_parser import pdf_to_images, process_image

from ai.groq_agents import cctv_expert, pa_expert, cost_expert, report_generator
from ai.yolo_detector import detect_elements

from cad.dxf_export import export_to_dxf, export_to_industrial_dxf
from cad.dxf_processor import DXFProcessor

from core.placement import add_door_cameras, get_building_mask, get_wall_mask, is_inside, away_from_wall, place_access_control
from core.optimization import optimize_coverage, create_mask_from_rooms
from core.sira_rules import apply_sira_compliance
# Corrected imports for industrial CAD and SIRA compliance

st.set_page_config(page_title="ELV AI System", layout="wide", page_icon="🏢")

st.title("🏢 ELV AI Design Assistant (PRO Architecture)")
st.markdown("---")

# INPUT SECTION
st.sidebar.header("📥 Building Requirements")

area = st.sidebar.number_input("Total Area (sqm)", min_value=50, value=500, step=50)
floors = st.sidebar.number_input("Number of Floors", min_value=1, value=1, step=1)
building_type = st.sidebar.selectbox(
    "Building Type",
    ["Commercial", "Residential", "Warehouse"]
)

st.sidebar.markdown("---")
st.sidebar.header("🚪 Master Topological Inputs")
manual_entry_points = st.sidebar.number_input("Entry Points (Main Doors)", min_value=1, value=2, step=1)
manual_staircases = st.sidebar.number_input("Staircases", min_value=0, value=1, step=1)
manual_rooms = st.sidebar.number_input("Estimated Total Rooms", min_value=1, value=10, step=1)

col_area1, col_area2 = st.sidebar.columns(2)
with col_area1:
    outside_area = st.number_input("Outside (sqm)", min_value=0, value=0, step=50)
with col_area2:
    inside_area = st.number_input("Inside (sqm)", min_value=50, value=500, step=50)

st.sidebar.markdown("---")
st.sidebar.header("🗺️ Blueprint Vision (YOLO)")
blueprint_file = st.sidebar.file_uploader("Upload Floor Plan (PDF/Image/DXF)", type=["png", "jpg", "jpeg", "pdf", "dxf"])
generate_simulated = st.sidebar.checkbox("Generate simulated layout if no file uploaded", value=False, help="Enable this to see a mathematical topological placement without uploading a floor plan.")

st.sidebar.header("📐 CAD Overlay (Optional)")
base_dxf_file = st.sidebar.file_uploader("Upload Base DXF File", type=["dxf"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Advanced Physical Parameters")
camera_fov = st.sidebar.slider("Camera FOV Radius (m)", 5, 30, 15, help="How far the camera can reliably detect motion.")
speaker_coverage = st.sidebar.slider("Speaker Acoustic Radius (m)", 3, 20, 8, help="How far the speaker's sound carries at acceptable dB.")
px_per_meter = st.sidebar.number_input("Pixels Per Meter Scale", min_value=1, value=20, step=1, help="Conversion scale of the floor plan.")

st.sidebar.markdown("---")
st.sidebar.header("👁️ System Export Filter")
system_view = st.sidebar.radio("Select systems to render & export:", ["All Systems", "Cameras Only", "Speakers Only", "Access Control Only"])

st.sidebar.markdown("---")
st.sidebar.header("💰 Hardware Pricing (AED)")
with st.sidebar.expander("Override Default Unit Prices"):
    price_cam = st.number_input("Camera Unit Price", min_value=0.0, value=250.0, step=10.0)
    price_spk = st.number_input("Speaker Unit Price", min_value=0.0, value=150.0, step=10.0)
    price_acs = st.number_input("Access Control Node", min_value=0.0, value=2900.0, step=100.0)
    price_cat6 = st.number_input("Cat6 Cable (per m)", min_value=0.0, value=4.50, step=0.5)
    price_audio = st.number_input("Audio Cable (per m)", min_value=0.0, value=3.00, step=0.5)
    price_acs_cab = st.number_input("ACS Cable (per m)", min_value=0.0, value=9.00, step=0.5)
    
prices_dict = {
    "camera": price_cam,
    "speaker": price_spk,
    "acs": price_acs,
    "cable_cctv": price_cat6,
    "cable_pa": price_audio,
    "cable_acs": price_acs_cab
}


if st.sidebar.button("Generate ELV Plan & Routing", use_container_width=True):
    with st.spinner("Analyzing building requirements and floor plans..."):
        
        # Initialize variables
        final_cameras = 0
        final_speakers = 0
        final_acs = 0
        img = None
        fov_image = None
        mask = None
        layout = {"cameras": [], "speakers": [], "cable_routes": []}
        actual_cameras = 0
        actual_speakers = 0
        vision_success = False
        is_dxf_mode = False
        processor = None

        # --- 1. VISION PIPELINE (Prioritized over manual input) ---
        if blueprint_file is not None:
            file_ext = blueprint_file.name.split('.')[-1].lower()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                tmp_path = tmp_file.name
            with open(tmp_path, "wb") as f:
                f.write(blueprint_file.getbuffer())
                
            # --- NEW INDUSTRIAL DXF WORKFLOW ---
            if file_ext == 'dxf':
                st.info("Parsing Industrial DXF Floor Plan...")
                processor = DXFProcessor(tmp_path)
                mask = processor.generate_building_mask()
                corridors = processor.get_corridor_zones()
                entries = processor.get_entry_points()
                area = processor.get_total_area() # Sync area for costing
                
                # SIRA Compliant Camera Placement
                cam_radius_px = int(camera_fov * 10) # 10 PPU
                cam_points = apply_sira_compliance(mask, corridors, entries, cam_radius_px)
                
                layout["cameras"] = [{"x": p[0], "y": p[1]} for p in cam_points]
                
                # Standard Speaker Placement
                spk_radius_px = int(speaker_coverage * 10)
                spk_points = optimize_coverage(mask, radius_px=spk_radius_px, coverage_threshold=0.98, edge_bias=False)
                layout["speakers"] = [{"x": p[0], "y": p[1]} for p in spk_points]
                
                final_cameras = len(cam_points)
                final_speakers = len(spk_points)
                final_acs = 0
                vision_success = True
                is_dxf_mode = True
            else:
                try:
                    if file_ext == "pdf":
                        pil_img = pdf_to_images(tmp_path)
                        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    else:
                        file_bytes = np.fromfile(tmp_path, np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        # Run YOLO Object Detection
                        detections = detect_elements(img)
                        mask = get_building_mask(img)
                        wall_mask = get_wall_mask(img)
                        detected_rooms = []
                        for r in detections.get("rooms", []) + detections.get("corridors", []):
                            x1, y1, x2, y2 = r["bbox"]
                            detected_rooms.append((x1, y1, x2 - x1, y2 - y1))
                        
                        # 2. Place cameras using mathematical Coverage Optimization (Set Cover)
                        cam_radius_px = int(camera_fov * px_per_meter)
                        cam_points = optimize_coverage(mask, radius_px=cam_radius_px, coverage_threshold=0.95, edge_bias=True)
                        doors = [d["bbox"] for d in detections.get("doors", [])]
                        cam_points += add_door_cameras(doors)
                        cam_points = [p for p in cam_points if away_from_wall(wall_mask, p)]
                        layout["cameras"] = [{"x": p[0], "y": p[1]} for p in cam_points]
                        
                        # --- NEW MATHEMATICAL SPEAKER ENGINE ---
                        spk_radius_px = int(speaker_coverage * px_per_meter)
                        spk_points = optimize_coverage(mask, radius_px=spk_radius_px, coverage_threshold=0.98, edge_bias=False)
                        layout["speakers"] = [{"x": p[0], "y": p[1]} for p in spk_points]
                        
                        # --- NEW ACCESS CONTROL ENGINE ---
                        acs_points = place_access_control(doors=doors, rooms=detected_rooms, entry_points=manual_entry_points, staircases=manual_staircases)
                        layout["access_control"] = [{"x": p[0], "y": p[1]} for p in acs_points]
                        
                        final_cameras = len(cam_points)
                        final_speakers = len(spk_points)
                        final_acs = len(acs_points)
                        vision_success = True
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                
            if os.path.exists(tmp_path) and not is_dxf_mode:
                os.remove(tmp_path)
            
        # --- 2. FALLBACK (If no image uploaded) ---
        if not vision_success:
            final_cameras = calculate_cameras(area)
            final_speakers = calculate_speakers(area)
            final_acs = calculate_access_control(manual_entry_points, manual_rooms, manual_staircases)
            
            if generate_simulated:
                st.info("No valid floor plan analyzed. Generating a simulated rough topological map based on area.")
                side_m = int(np.sqrt(area))
                side_px = int(side_m * px_per_meter)
                img_size = side_px + 100
                img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
                cv2.rectangle(img, (50, 50), (50 + side_px, 50 + side_px), (200, 200, 200), -1)
                simulated_room = [(50, 50, side_px, side_px)]
                sim_mask = create_mask_from_rooms(simulated_room, img_size, img_size)
                cam_radius_px = int(camera_fov * px_per_meter)
                cam_points = optimize_coverage(sim_mask, radius_px=cam_radius_px, coverage_threshold=0.95, edge_bias=True)
                layout["cameras"] = [{"x": p[0], "y": p[1]} for p in cam_points[:final_cameras]]
                spk_radius_px = int(speaker_coverage * px_per_meter)
                spk_points = optimize_coverage(sim_mask, radius_px=spk_radius_px, coverage_threshold=0.98, edge_bias=False)
                layout["speakers"] = [{"x": p[0], "y": p[1]} for p in spk_points[:final_speakers]]
                acs_points = place_access_control(doors=[], rooms=simulated_room, entry_points=manual_entry_points, staircases=manual_staircases)
                layout["access_control"] = [{"x": p[0], "y": p[1]} for p in acs_points[:final_acs]]
                vision_success = True

        # --- 2.5 RENDER HARDWARE LAYOUT ---
        if vision_success:
            if is_dxf_mode:
                # Create a viewable image from the DXF processor's data
                fov_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                fov_image = cv2.bitwise_not(fov_image) # White background
            else:
                fov_image = img.copy()
                
            routes = []
            
            # 1. Cameras
            if system_view in ["All Systems", "Cameras Only"]:
                for p in layout.get("cameras", []):
                    cv2.circle(fov_image, (p["x"], p["y"]), 8, (0, 0, 255), -1)
                
            # 2. Speakers
            if system_view in ["All Systems", "Speakers Only"]:
                for p in layout.get("speakers", []):
                    cv2.circle(fov_image, (p["x"], p["y"]), 6, (255, 0, 0), -1)
                    
            # 3. Access Control
            if system_view in ["All Systems", "Access Control Only"]:
                for p in layout.get("access_control", []):
                    pt = (p["x"], p["y"])
                    cv2.rectangle(fov_image, (pt[0]-6, pt[1]-6), (pt[0]+6, pt[1]+6), (255, 0, 255), -1)
            
            layout["cable_routes"] = routes

        # --- 3. CORE CALCULATIONS ---
        cable_lengths = calculate_cable_length(final_cameras, final_speakers, final_acs)
        cost_breakdown = calculate_cost(final_cameras, final_speakers, final_acs, cable_lengths, prices_dict)
        grand_total = cost_breakdown["grand_total"]
        final_cable = cost_breakdown["cables"]["total_length"]

        # --- 4. MULTI-AGENT ANALYSIS ---
        cctv_text = cctv_expert(area, final_cameras, building_type, inside_area, outside_area)
        pa_text = pa_expert(area, final_speakers, building_type)
        cost_text = cost_expert(grand_total)

    # ================== UI RENDERING ==================

    st.header("📊 Finalized Hardware Requirements")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📷 Cameras", final_cameras)
    with col2:
        st.metric("🔊 Speakers", final_speakers)
    with col3:
        st.metric("🚪 Access Control", final_acs)
    with col4:
        st.metric("🔌 Cable Length (m)", final_cable)
    with col5:
        st.metric("💰 Total Cost", f"AED {grand_total:,.2f}")

    # --- 4. PROFESSIONAL DASHBOARD ---
    st.markdown("---")
    
    if vision_success:
        st.header("📡 Pro Visual Layout Routing")
        st.image(fov_image, channels="BGR", caption=f"Engineering Preview ({'Industrial DXF' if is_dxf_mode else 'Vision Model'})")
        
        st.markdown("---")
        st.header("💾 Export Architecture Models")
        base_dxf_path = None
        if base_dxf_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp_dxf:
                tmp_dxf.write(base_dxf_file.read())
                base_dxf_path = tmp_dxf.name
        
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            if is_dxf_mode:
                output_dxf = f"SIRA_Design_{blueprint_file.name}"
                export_to_industrial_dxf(layout, output_dxf, base_dxf_path=tmp_path, processor=processor)
            else:
                output_dxf = "project_elv_output.dxf"
                export_to_industrial_dxf(layout, output_dxf)
            
            with open(output_dxf, "rb") as f:
                st.download_button("📂 Download SIRA-Compliant DXF", f, file_name=output_dxf)
            if base_dxf_path and os.path.exists(base_dxf_path):
                os.remove(base_dxf_path)
                
        with col_export2:
            json_str = json.dumps(layout, indent=4)
            
        st.markdown("---")

    # COST TABLE UI
    st.header("🛒 Cost Estimation Breakdown")
    cost_df = pd.DataFrame([
        {"Component": "📷 CCTV Cameras", "Quantity": cost_breakdown["cameras"]["quantity"], "Unit Price": f"AED {cost_breakdown['cameras']['unit_price']:,.2f}", "Total": f"AED {cost_breakdown['cameras']['total']:,.2f}"},
        {"Component": "🔊 PA Speakers", "Quantity": cost_breakdown["speakers"]["quantity"], "Unit Price": f"AED {cost_breakdown['speakers']['unit_price']:,.2f}", "Total": f"AED {cost_breakdown['speakers']['total']:,.2f}"},
        {"Component": "🚪 Access Control", "Quantity": cost_breakdown["access_control"]["quantity"], "Unit Price": f"AED {cost_breakdown['access_control']['unit_price']:,.2f}", "Total": f"AED {cost_breakdown['access_control']['total']:,.2f}"},
        {"Component": "🔌 Cat6 Cable (CCTV)", "Quantity": f"{cost_breakdown['cables']['cctv']['quantity']}m", "Unit Price": f"AED {cost_breakdown['cables']['cctv']['unit_price']:,.2f}/m", "Total": f"AED {cost_breakdown['cables']['cctv']['total']:,.2f}"},
        {"Component": "🔌 18AWG Audio Cable (PA)", "Quantity": f"{cost_breakdown['cables']['pa']['quantity']}m", "Unit Price": f"AED {cost_breakdown['cables']['pa']['unit_price']:,.2f}/m", "Total": f"AED {cost_breakdown['cables']['pa']['total']:,.2f}"},
        {"Component": "🔌 Composite Cable (ACS)", "Quantity": f"{cost_breakdown['cables']['acs']['quantity']}m", "Unit Price": f"AED {cost_breakdown['cables']['acs']['unit_price']:,.2f}/m", "Total": f"AED {cost_breakdown['cables']['acs']['total']:,.2f}"},
    ])
    st.table(cost_df)
    
    st.markdown("---")

    # MULTI-AGENT UI
    st.header("📄 Specialist AI Reports")
    tab1, tab2, tab3 = st.tabs(["📷 CCTV Architect", "🔊 Acoustics Engineer", "💵 Cost Estimator"])
    
    with tab1:
        st.markdown(cctv_text)
    with tab2:
        st.markdown(pa_text)
    with tab3:
        st.markdown(cost_text)