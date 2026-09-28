"""AeroFuel AI - Aviation Domain Constants, Envelopes, and Safety Disclaimers.

Contains physics constants, ICAO aircraft type classifications, emissions factors,
and operational bounds derived from public aviation literature and EUROCONTROL benchmarks.
"""

from typing import Dict, List, Set

# Non-operational safety disclaimer required on all analytical outputs
SYSTEM_DISCLAIMER = (
    "This system is an analytical decision-support prototype. It does not provide "
    "operational flight-planning or safety-critical recommendations. Actual airline "
    "fuel planning must follow approved aircraft performance data, operational procedures, "
    "dispatch requirements, regulatory requirements, and flight crew/operations-control decisions."
)

# Standard Emissions Factors
# ICAO Doc 9889 / EUROCONTROL: 3.15 - 3.16 kg CO2 per kg Jet-A1 fuel
CO2_PER_KG_FUEL: float = 3.16

# Atmospheric & Physics Constants (ISA standard)
ISA_SEA_LEVEL_PRESSURE_HPA: float = 1013.25
ISA_SEA_LEVEL_TEMP_K: float = 288.15
GRAVITY_ACCELERATION: float = 9.80665
AIR_GAS_CONSTANT: float = 287.05
METERS_PER_FOOT: float = 0.3048
FEET_PER_METER: float = 3.28084
KNOTS_PER_MPS: float = 1.94384
MPS_PER_KNOT: float = 0.514444
NM_PER_METER: float = 0.000539957

# Fuel properties (Jet A-1)
JET_A1_DENSITY_KG_L: float = 0.804  # ~0.804 kg per liter at 15 deg C
JET_A1_LHV_MJ_KG: float = 43.15     # Lower Heating Value (MJ/kg)

# Aircraft Type Categories (EUROCONTROL PRC 2025 Data Challenge - 27 Types)
AIRCRAFT_CATEGORIES: Dict[str, str] = {
    # Narrow-body
    "A20N": "Narrow-body",
    "A21N": "Narrow-body",
    "A318": "Narrow-body",
    "A319": "Narrow-body",
    "A320": "Narrow-body",
    "A321": "Narrow-body",
    "B38M": "Narrow-body",
    "B39M": "Narrow-body",
    "B737": "Narrow-body",
    "B738": "Narrow-body",
    "B739": "Narrow-body",
    "B752": "Narrow-body",
    # Wide-body
    "A306": "Wide-body",
    "A332": "Wide-body",
    "A333": "Wide-body",
    "A359": "Wide-body",
    "A388": "Wide-body",
    "B744": "Wide-body",
    "B748": "Wide-body",
    "B763": "Wide-body",
    "B772": "Wide-body",
    "B77L": "Wide-body",
    "B77W": "Wide-body",
    "B788": "Wide-body",
    "B789": "Wide-body",
    # Freighters / Tri-jets
    "MD11": "Freighter",
}

# Aircraft Families for Grouping & Cross-Type Generalization
AIRCRAFT_FAMILIES: Dict[str, str] = {
    "A20N": "A320neo",
    "A21N": "A320neo",
    "A318": "A320ceo",
    "A319": "A320ceo",
    "A320": "A320ceo",
    "A321": "A320ceo",
    "B38M": "B737MAX",
    "B39M": "B737MAX",
    "B737": "B737NG",
    "B738": "B737NG",
    "B739": "B737NG",
    "B752": "B757",
    "A306": "A300",
    "A332": "A330",
    "A333": "A330",
    "A359": "A350",
    "A388": "A380",
    "B744": "B747",
    "B748": "B747",
    "B763": "B767",
    "B772": "B777",
    "B77L": "B777",
    "B77W": "B777",
    "B788": "B787",
    "B789": "B787",
    "MD11": "MD-11",
}

# Reference Nominal Cruise Envelopes (for constrained optimization & realistic bounds)
# Altitude in feet, Speed in Mach and Groundspeed knots
AIRCRAFT_ENVELOPES: Dict[str, Dict[str, float]] = {
    "A320": {"min_fl": 290, "max_fl": 390, "opt_fl": 350, "min_mach": 0.74, "max_mach": 0.82, "nominal_mach": 0.78, "ref_mass_kg": 64000.0},
    "A20N": {"min_fl": 290, "max_fl": 390, "opt_fl": 360, "min_mach": 0.74, "max_mach": 0.82, "nominal_mach": 0.78, "ref_mass_kg": 65000.0},
    "A321": {"min_fl": 290, "max_fl": 390, "opt_fl": 350, "min_mach": 0.74, "max_mach": 0.82, "nominal_mach": 0.78, "ref_mass_kg": 73000.0},
    "A21N": {"min_fl": 290, "max_fl": 390, "opt_fl": 360, "min_mach": 0.74, "max_mach": 0.82, "nominal_mach": 0.78, "ref_mass_kg": 75000.0},
    "B738": {"min_fl": 290, "max_fl": 410, "opt_fl": 360, "min_mach": 0.74, "max_mach": 0.82, "nominal_mach": 0.78, "ref_mass_kg": 65000.0},
    "B38M": {"min_fl": 290, "max_fl": 410, "opt_fl": 370, "min_mach": 0.74, "max_mach": 0.82, "nominal_mach": 0.78, "ref_mass_kg": 66000.0},
    "B77W": {"min_fl": 290, "max_fl": 430, "opt_fl": 340, "min_mach": 0.80, "max_mach": 0.86, "nominal_mach": 0.84, "ref_mass_kg": 260000.0},
    "A359": {"min_fl": 310, "max_fl": 430, "opt_fl": 380, "min_mach": 0.81, "max_mach": 0.87, "nominal_mach": 0.85, "ref_mass_kg": 200000.0},
    "B789": {"min_fl": 310, "max_fl": 430, "opt_fl": 380, "min_mach": 0.81, "max_mach": 0.87, "nominal_mach": 0.85, "ref_mass_kg": 205000.0},
    "A388": {"min_fl": 290, "max_fl": 430, "opt_fl": 350, "min_mach": 0.81, "max_mach": 0.87, "nominal_mach": 0.85, "ref_mass_kg": 430000.0},
}

# Default envelope fallback for other types
DEFAULT_ENVELOPE = {
    "min_fl": 280,
    "max_fl": 410,
    "opt_fl": 350,
    "min_mach": 0.74,
    "max_mach": 0.84,
    "nominal_mach": 0.78,
    "ref_mass_kg": 70000.0,
}

# Flight Phase Definitions based on Kinematic Thresholds
PHASE_CLIMB = "Climb"
PHASE_CRUISE = "Cruise"
PHASE_DESCENT = "Descent"
PHASE_APPROACH = "Approach"
PHASE_UNKNOWN = "Unknown"

# Default Fuel Price for Cost Modeling (USD per kg)
# ~0.75 - 0.90 USD/kg (~$2.30 - $2.80 per US gallon of Jet A-1)
DEFAULT_FUEL_PRICE_USD_PER_KG: float = 0.82
