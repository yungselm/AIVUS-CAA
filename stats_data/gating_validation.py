import os
import pandas as pd

# Constants
WORKDIR = r'D:/NIVA_analysis/NIVA_analysis'
INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
# Registration thresholds for each modality
REGISTRATION = {
    'ivus_rest': 49,
    'ivus_stress': 54,
    'fl_rest': 13,
    'fl_stress': 6
}

# File mappings
FILES = {
    'ivus_rest': 'Run4-GatingCO5BP139_78.txt',
    'ivus_stress': 'Run15-GatingCO15BP187_92RCA240.txt',
    'ivus_rest_auto': 'Run4-GatingCO5BP139_78automatic.txt',
    'ivus_stress_auto': 'Run15-GatingCO15BP187_92RCA240automatic.txt',
    'eye_rest': 'eye_PDD6U1UJ_report.txt',
    'eye_stress': 'eye_PD33CYTJ_report.txt',
    'fl_rest': 'narco_253_phantom_pressure_rest_run03.csv',
    'fl_stress': 'narco_253_phantom_pressure_rest_run15.csv'
}

# Ensure output directory exists and set working directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(WORKDIR)

# Helper functions
def load_table(name):
    """Load a table by name with appropriate delimiter."""
    path = os.path.join(INPUT_DIR, FILES[name])
    sep = '\t' if name.startswith(('ivus', 'eye')) else ','
    return pd.read_csv(path, sep=sep)


def get_registration(name):
    """Determine the registration threshold based on name prefix."""
    if name.startswith('ivus_rest') or name.startswith('eye_rest'):
        return REGISTRATION['ivus_rest']
    if name.startswith('ivus_stress') or name.startswith('eye_stress'):
        return REGISTRATION['ivus_stress']
    if name.startswith('fl_rest'):
        return REGISTRATION['fl_rest']
    if name.startswith('fl_stress'):
        return REGISTRATION['fl_stress']
    raise KeyError(f"No registration defined for {name}")


def register_and_trim(df, name):
    """Trim frames/time before registration threshold."""
    reg = get_registration(name)
    key = 'frame' if name.startswith(('ivus', 'eye')) else 'time'
    return df[df[key] >= reg].copy()


def drop_invalid(df, name):
    """Drop invalid rows based on modality."""
    if name.startswith(('ivus', 'eye')):
        return df[df['phase'] != '-']
    return df[df['peaks'] != 0]


def add_phase(df, name):
    """Add a 'phase' column for FL data."""
    if name.startswith('fl'):
        df['phase'] = df['peaks'].map({2: 'D'}).fillna('S')
    return df


def compute_frame(df, name):
    """Compute a zero-based 'frame' column for each dataset."""
    if name.startswith('fl'):
        # Convert time to frames at 30 fps, drop first row
        df['frame'] = ((df['time'] - df['time'].iloc[0]) * 30).round().astype(int)
        return df.iloc[1:].reset_index(drop=True)
    # For IVUS/eye: drop first row and reset frame to zero
    df = df.iloc[1:].reset_index(drop=True)
    df['frame'] = df['frame'] - df['frame'].iloc[0]
    return df


def process(name):
    """Load, register, clean, and compute frame for a dataset."""
    df = load_table(name)
    df = register_and_trim(df, name)
    df = drop_invalid(df, name)
    df = add_phase(df, name)
    df = compute_frame(df, name)
    return df

# Process all datasets
data = {name: process(name) for name in FILES}

# Function to save ICC tables

def save_icc(df1, df2, col1, col2, out_name):
    icc = pd.DataFrame({
        'phase': df1['phase'],
        col1: df1[col2].astype(int),
        col2: df2[col2]
    }).dropna()
    icc.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)

# Generate ICC outputs
save_icc(data['ivus_rest'], data['fl_rest'], 'ivus_frame', 'frame', 'icc_gating_rest.csv')
save_icc(data['ivus_stress'], data['fl_stress'], 'ivus_frame', 'frame', 'icc_gating_stress.csv')
save_icc(data['ivus_rest_auto'], data['ivus_rest'], 'ivus_frame_auto', 'frame', 'icc_auto_manual_rest.csv')
save_icc(data['ivus_stress_auto'], data['ivus_stress'], 'ivus_frame_auto', 'frame', 'icc_auto_manual_stress.csv')
save_icc(data['eye_rest'], data['fl_rest'], 'ivus_frame', 'frame', 'icc_gating_rest_eye.csv')
save_icc(data['eye_stress'], data['fl_stress'], 'ivus_frame', 'frame', 'icc_gating_stress_eye.csv')
