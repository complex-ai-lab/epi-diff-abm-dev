#!/usr/bin/env python3
"""
build_presentation.py

Generates a PowerPoint presentation showing simulation results and calibrated parameters
for all 153 counties hardcoded in run_all_sims.sh.
"""

import os
import sys
import glob
import yaml
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# Hardcoded list of all 153 counties from run_all_sims.sh
COUNTIES = [
    '01009', '01031', '01039', '01045', '01049', '01071', '13013', '13071', '13103', '13115',
    '13137', '13153', '17011', '17021', '17037', '17055', '17063', '17073', '17091', '17095',
    '17103', '17115', '17117', '17121', '17141', '17167', '19017', '19027', '19113', '19155',
    '19169', '20035', '20045', '20061', '20079', '20103', '20125', '20149', '20161', '20169',
    '21009', '21029', '21035', '21071', '21073', '21083', '21089', '21093', '21113', '21179',
    '21199', '21209', '21211', '22001', '22005', '22039', '22063', '26023', '26027', '26041',
    '26057', '26059', '26067', '26073', '26123', '27005', '27027', '27041', '27059', '27085',
    '27109', '29021', '29027', '29051', '29101', '30029', '30047', '30049', '30093', '36003',
    '36011', '36013', '36031', '36051', '38059', '38077', '38093', '39005', '39007', '39011',
    '39013', '39015', '39021', '39023', '39031', '39033', '39037', '39039', '39045', '39051',
    '39057', '39059', '39063', '39071', '39077', '39079', '39083', '39087', '40021', '40089',
    '40097', '40121', '40147', '42005', '42013', '42025', '42031', '42051', '42055', '42059',
    '42061', '42075', '42085', '42087', '46011', '46013', '46029', '46035', '46081', '46083',
    '46135', '47003', '47005', '47009', '47017', '48001', '48005', '48013', '48041', '48049',
    '48071', '48091', '48097', '48143', '48147', '48181', '48189', '48213', '51047', '51095',
    '51137', '51149', '51161'
]

# Standard state FIPS dictionary mapping
STATE_FIPS_MAP = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT', '10': 'DE',
    '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA',
    '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN',
    '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM',
    '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
    '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA',
    '54': 'WV', '55': 'WI', '56': 'WY'
}

def load_county_names():
    """Builds a mapping from 5-digit FIPS to (County Name, State Abbreviation)."""
    fips_map = {}
    policy_db_path = os.path.join("data", "county_policy_data", "39109-0001-Data.tsv")
    if os.path.exists(policy_db_path):
        try:
            df = pd.read_csv(policy_db_path, sep="\t", usecols=["COUNTY_FIPS", "COUNTYNAME", "STATE"])
            df = df.dropna(subset=["COUNTY_FIPS"]).drop_duplicates(subset=["COUNTY_FIPS"])
            for _, row in df.iterrows():
                fips_str = str(int(row["COUNTY_FIPS"])).zfill(5)
                fips_map[fips_str] = (str(row["COUNTYNAME"]).strip(), str(row["STATE"]).strip())
        except Exception as e:
            print(f"Warning loading policy dataset for county names: {e}")

    return fips_map

def load_config():
    """Loads configuration metadata from covid_abm/yamls/config.yaml or covid_abm/config.yaml."""
    config_path = os.path.join("covid_abm", "yamls", "config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join("covid_abm", "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError("Could not locate config.yaml in covid_abm/yamls/ or covid_abm/.")
        
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_county_folder_and_graph(population, config):
    """
    Constructs the config directory for a county and finds the latest epoch's simulation_results.png
    and calibrated_params.txt.
    """
    meta = config.get("simulation_metadata", {})
    date = meta.get("DATE", "202010-202104")
    initial_rate = meta.get("INITIAL_INFECTION_RATE", 0.0005)
    exposed_to_infected = meta.get("EXPOSED_TO_INFECTED_TIME", 3)
    infected_to_recovered = meta.get("INFECTED_TO_RECOVERED_TIME", 5)
    with_k = meta.get("WITH_K", True)
    with_vacc = meta.get("WITH_VACC", False)
    use_7day_avg = meta.get("USE_7DAY_AVG", True)
    metro_phase = meta.get("metro_calibration_phase", 0)

    # Reconstruct folder string based on abm_nets.py
    folder_pattern = f"{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}_{use_7day_avg}_metro_{metro_phase}"
    base_dir = os.path.join("result_graphs", population, date, folder_pattern)

    # Check if exact directory exists, or search for potential metro phase variations
    if not os.path.exists(base_dir):
        # Fallback search if metro_phase differed
        matches = glob.glob(os.path.join("result_graphs", population, date, f"{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_*"))
        if matches:
            base_dir = sorted(matches)[-1]

    if not os.path.exists(base_dir):
        return None, None, None

    calibrated_params_file = os.path.join(base_dir, "calibrated_params.txt")

    # Find highest numeric epoch directory with simulation_results.png
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    subdirs = sorted(subdirs, key=lambda x: int(x), reverse=True)

    graph_path = None
    latest_epoch = None

    for epoch_dir in subdirs:
        img_path = os.path.join(base_dir, epoch_dir, "simulation_results.png")
        if os.path.exists(img_path):
            graph_path = img_path
            latest_epoch = epoch_dir
            break

    return base_dir, graph_path, calibrated_params_file

def parse_calibrated_params(params_file):
    """
    Reads calibrated_params.txt and extracts Average R2, Initial Infection Rate, and K parameter.
    """
    if not params_file or not os.path.exists(params_file):
        return None

    try:
        data = np.loadtxt(params_file)
        if data.ndim == 0 or len(data) < 3:
            return None

        weekly_r2 = data[:-2]
        avg_r2 = float(np.mean(weekly_r2))
        initial_rate = float(data[-2])
        k_param = float(data[-1])

        return {
            "avg_r2": avg_r2,
            "initial_rate": initial_rate,
            "k_param": k_param,
            "num_weeks": len(weekly_r2)
        }
    except Exception as e:
        print(f"Error parsing calibrated params from {params_file}: {e}")
        return None

def create_presentation(output_pptx="simulation_results_presentation.pptx"):
    """Builds the full presentation for all counties."""
    config = load_config()
    county_name_lookup = load_county_names()

    prs = Presentation()
    # Set 16:9 widescreen layout
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]  # Blank slide layout

    print(f"Generating presentation for {len(COUNTIES)} counties...")

    for idx, fips in enumerate(COUNTIES, 1):
        # Resolve county & state name
        if fips in county_name_lookup:
            c_name, state_abbr = county_name_lookup[fips]
            title_text = f"{c_name}, {state_abbr} (FIPS: {fips})"
        else:
            state_abbr = STATE_FIPS_MAP.get(fips[:2], "US")
            title_text = f"County {fips}, {state_abbr} (FIPS: {fips})"

        slide = prs.slides.add_slide(blank_layout)

        # -------------------------------------------------------------
        # Slide Header (Top Left)
        # -------------------------------------------------------------
        title_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12.0), Inches(0.8))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = RGBColor(27, 38, 59)

        # -------------------------------------------------------------
        # Left Side: Fitting Graph (simulation_results.png)
        # -------------------------------------------------------------
        base_dir, graph_path, params_file = get_county_folder_and_graph(fips, config)

        if graph_path and os.path.exists(graph_path):
            slide.shapes.add_picture(graph_path, Inches(0.6), Inches(1.4), width=Inches(7.2))
        else:
            # Placeholder box if graph missing
            shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.4), Inches(7.2), Inches(5.4))
            shape.fill.solid()
            shape.fill.fore_color.rgb = RGBColor(240, 240, 240)
            shape.line.color.rgb = RGBColor(200, 200, 200)
            tf_g = shape.text_frame
            p_g = tf_g.paragraphs[0]
            p_g.text = f"Fitting Graph (simulation_results.png)\nNot Found for {fips}"
            p_g.alignment = PP_ALIGN.CENTER
            p_g.font.size = Pt(18)
            p_g.font.color.rgb = RGBColor(120, 120, 120)

        # -------------------------------------------------------------
        # Right Side: Calibrated Parameters Card
        # -------------------------------------------------------------
        card_left = Inches(8.2)
        card_top = Inches(1.4)
        card_width = Inches(4.5)
        card_height = Inches(5.4)

        # Background card container
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, card_left, card_top, card_width, card_height)
        card.fill.solid()
        card.fill.fore_color.rgb = RGBColor(248, 249, 250)
        card.line.color.rgb = RGBColor(220, 224, 230)
        card.line.width = Pt(1.5)

        # Card Title
        card_title_box = slide.shapes.add_textbox(card_left + Inches(0.2), card_top + Inches(0.2), card_width - Inches(0.4), Inches(0.6))
        tf_ct = card_title_box.text_frame
        tf_ct.word_wrap = True
        p_ct = tf_ct.paragraphs[0]
        p_ct.text = "Final Calibrated Parameters"
        p_ct.font.size = Pt(20)
        p_ct.font.bold = True
        p_ct.font.color.rgb = RGBColor(13, 27, 42)

        # Parse Parameters
        params = parse_calibrated_params(params_file)

        param_box = slide.shapes.add_textbox(card_left + Inches(0.2), card_top + Inches(0.9), card_width - Inches(0.4), card_height - Inches(1.1))
        tf_p = param_box.text_frame
        tf_p.word_wrap = True

        if params:
            lines = [
                ("Average R₂", f"{params['avg_r2']:.4f}"),
                ("Initial Infection Rate", f"{params['initial_rate']:.6f}"),
                ("K Parameter (Dispersion)", f"{params['k_param']:.4f}"),
                ("Evaluated Weeks", f"{params['num_weeks']} weeks")
            ]

            for i, (label, val) in enumerate(lines):
                p_label = tf_p.paragraphs[0] if i == 0 else tf_p.add_paragraph()
                p_label.text = label
                p_label.font.size = Pt(14)
                p_label.font.bold = True
                p_label.font.color.rgb = RGBColor(65, 90, 119)
                p_label.space_after = Pt(2)
                p_label.space_before = Pt(10) if i > 0 else Pt(0)

                p_val = tf_p.add_paragraph()
                p_val.text = val
                p_val.font.size = Pt(20)
                p_val.font.bold = True
                p_val.font.color.rgb = RGBColor(27, 38, 59)
                p_val.space_after = Pt(8)
        else:
            p_err = tf_p.paragraphs[0]
            p_err.text = "Calibrated parameters (calibrated_params.txt)\nnot found or incomplete."
            p_err.font.size = Pt(14)
            p_err.font.italic = True
            p_err.font.color.rgb = RGBColor(180, 50, 50)

        if idx % 20 == 0 or idx == len(COUNTIES):
            print(f"Processed {idx}/{len(COUNTIES)} county slides...")

    prs.save(output_pptx)
    print(f"\nSuccessfully generated presentation: {os.path.abspath(output_pptx)}")

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "simulation_results_presentation.pptx"
    create_presentation(out_file)
