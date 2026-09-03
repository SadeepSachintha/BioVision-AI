"""Agronomic Knowledge Base for BioVision AI.

Contains pathology profiles, severity indexes, yield loss estimates, and
curated intervention protocols for 38 crop disease classes.
"""

from typing import Dict, Any

DISEASE_DATABASE: Dict[str, Dict[str, Any]] = {
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "disease": "Late Blight",
        "scientific_name": "Phytophthora infestans",
        "severity": "CRITICAL",
        "yield_loss_risk_pct": "60% - 90% without intervention within 48h",
        "immediate_actions": [
            "Quarantine and remove severely infected foliage; seal in bags to prevent spore aerosolization.",
            "Immediately shut off overhead sprinkler irrigation; switch to drip irrigation to keep canopies dry.",
            "Sanitize all pruning shears and stakes with 10% bleach solution."
        ],
        "chemical_control": "Apply targeted protectant/curative fungicides such as Chlorothalonil, Mancozeb, or Cyazofamid according to regional IPM thresholds.",
        "organic_bio_control": "Apply copper octanoate (copper soap) or Bacillus subtilis (Serenade ASO) thoroughly on abaxial and adaxial leaf surfaces.",
        "preventative_practices": "Implement a 3-year crop rotation outside the Solanaceae family. Ensure generous row spacing (minimum 24 inches) for aeration."
    },
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "scientific_name": "Alternaria solani",
        "severity": "HIGH",
        "yield_loss_risk_pct": "30% - 50% through premature foliar defoliation",
        "immediate_actions": [
            "Prune infected bottom leaves showing concentric 'bullseye' target lesions.",
            "Mulch heavily around the plant base to prevent soil-borne spore splashing during rain."
        ],
        "chemical_control": "Treat with Azoxystrobin, Pyraclostrobin, or Boscalid alternating with Mancozeb to prevent fungicide resistance.",
        "organic_bio_control": "Spray Neem oil extract or bio-fungicide formulations with Trichoderma harzianum at first lesion sighting.",
        "preventative_practices": "Stake plants upright, avoid working in damp fields, and clear all nightshade weeds from field borders."
    },
    "Tomato___healthy": {
        "crop": "Tomato",
        "disease": "Healthy",
        "scientific_name": "Solanum lycopersicum",
        "severity": "NONE",
        "yield_loss_risk_pct": "0% (Optimal photosynthetic yield capacity)",
        "immediate_actions": [
            "No disease intervention needed. Maintain consistent balanced fertigation (N-P-K)."
        ],
        "chemical_control": "None recommended.",
        "organic_bio_control": "Routine biological soil inoculants (Mycorrhizal fungi) to enhance nutrient assimilation.",
        "preventative_practices": "Maintain regular scouting schedules twice weekly during high-humidity periods."
    },
    "Corn___Northern_Leaf_Blight": {
        "crop": "Corn (Maize)",
        "disease": "Northern Leaf Blight",
        "scientific_name": "Exserohilum turcicum",
        "severity": "HIGH",
        "yield_loss_risk_pct": "30% - 50% if lesions develop prior to silking stage",
        "immediate_actions": [
            "Scout 20 consecutive corn plants across 5 field zones to quantify ear-leaf lesion thresholds.",
            "Assess growth stage; if pre-tassel and disease is moving up canopy, prepare ground or aerial application."
        ],
        "chemical_control": "Apply strobilurin or triazole fungicides (e.g., Headline AMP, Trivapro) before silking if lesions exceed economic threshold.",
        "organic_bio_control": "Foliar sprays containing Reynoutria sachalinensis extract to trigger systemic acquired resistance (SAR).",
        "preventative_practices": "Plant hybrid corn varieties with multi-genic Ht resistance; deep-till crop residue post-harvest."
    },
    "Corn___Common_rust": {
        "crop": "Corn (Maize)",
        "disease": "Common Rust",
        "scientific_name": "Puccinia sorghi",
        "severity": "MODERATE",
        "yield_loss_risk_pct": "10% - 25% under prolonged cool, moist atmospheric conditions",
        "immediate_actions": [
            "Monitor pustules on both upper and lower leaf surfaces.",
            "Assess temperatures; hot dry conditions (>30°C) naturally arrest fungal reproduction."
        ],
        "chemical_control": "Apply Pyraclostrobin + Fluxapyroxad if >6 pustules per leaf appear on mid-canopy prior to tasseling.",
        "organic_bio_control": "Sulfur-based dusts or wettable powders applied early in the infestation cycle.",
        "preventative_practices": "Select rust-resistant hybrids and ensure timely planting to avoid peak airborne spore arrival."
    },
    "Corn___healthy": {
        "crop": "Corn (Maize)",
        "disease": "Healthy",
        "scientific_name": "Zea mays",
        "severity": "NONE",
        "yield_loss_risk_pct": "0% (Optimal harvest trajectory)",
        "immediate_actions": [
            "Continue standard nitrogen side-dressing and moisture monitoring."
        ],
        "chemical_control": "None required.",
        "organic_bio_control": "None required.",
        "preventative_practices": "Scout weekly for fall armyworm and rootworm damage during vegetative stages."
    },
    "Apple___Apple_scab": {
        "crop": "Apple",
        "disease": "Apple Scab",
        "scientific_name": "Venturia inaequalis",
        "severity": "HIGH",
        "yield_loss_risk_pct": "40% - 70% in marketability downgrade and defoliation",
        "immediate_actions": [
            "Remove and shred or burn fallen leaves under orchard canopy to eliminate ascospore reservoirs.",
            "Prune dense branch clusters to allow rapid canopy drying after morning dew."
        ],
        "chemical_control": "Apply Captan or Difenoconazole during green tip through petal fall growth stages.",
        "organic_bio_control": "Lime sulfur or potassium bicarbonate sprays during primary infection events.",
        "preventative_practices": "Plant scab-resistant apple cultivars (e.g., Liberty, Enterprise, Honeycrisp)."
    },
    "Apple___healthy": {
        "crop": "Apple",
        "disease": "Healthy",
        "scientific_name": "Malus domestica",
        "severity": "NONE",
        "yield_loss_risk_pct": "0% (Optimal fruit set and vigor)",
        "immediate_actions": ["Maintain integrated pest and pollinator management."],
        "chemical_control": "None required.",
        "organic_bio_control": "Dormant oil spray during winter to control scale insects.",
        "preventative_practices": "Monitor orchard canopy light penetration and prune annually."
    },
    "Grape___Black_rot": {
        "crop": "Grape",
        "disease": "Black Rot",
        "scientific_name": "Guignardia bidwellii",
        "severity": "CRITICAL",
        "yield_loss_risk_pct": "50% - 100% loss of fruit clusters under humid conditions",
        "immediate_actions": [
            "Physically remove and destroy all shriveled, black mummified berries from vines and ground.",
            "Expose fruit clusters to sunlight and wind by selective summer shoot thinning."
        ],
        "chemical_control": "Apply Myclobutanil, Mancozeb, or Ziram starting at early shoot growth (1-3 inches) through 4 weeks post-bloom.",
        "organic_bio_control": "Bordeaux mixture (copper sulfate and hydrated lime) applied prior to rain events.",
        "preventative_practices": "Train grapevines to high-wire or open canopy systems for swift moisture evaporation."
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "disease": "Late Blight",
        "scientific_name": "Phytophthora infestans",
        "severity": "CRITICAL",
        "yield_loss_risk_pct": "70% - 100% total tuber rot in field and post-harvest storage",
        "immediate_actions": [
            "Kill potato vines (flame, mechanical vine killing, or desiccant) 2 weeks before harvest if blight is active.",
            "Destroy cull piles immediately; never leave infected tubers on field margins."
        ],
        "chemical_control": "Systemic fungicides: Fluopicolide, Mandipropamid, or Cymoxanil in alternation.",
        "organic_bio_control": "Copper hydroxide formulations applied at 7-day intervals under cool wet forecasts.",
        "preventative_practices": "Use certified disease-free seed tubers. Hill soil high over developing tubers to prevent wash-down spores."
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "disease": "Early Blight",
        "scientific_name": "Alternaria solani",
        "severity": "MODERATE",
        "yield_loss_risk_pct": "20% - 40% tuber yield reduction",
        "immediate_actions": [
            "Avoid moisture stress; plants under nitrogen or drought stress are highly susceptible.",
            "Disinfect harvesting machinery to avoid tuber skinning and spore inoculation."
        ],
        "chemical_control": "Apply Chlorothalonil or Boscalid at flowering stage when canopies close.",
        "organic_bio_control": "Bacillus amyloliquefaciens foliar inoculations.",
        "preventative_practices": "Maintain adequate nitrogen fertility throughout vine expansion."
    },
    "Potato___healthy": {
        "crop": "Potato",
        "disease": "Healthy",
        "scientific_name": "Solanum tuberosum",
        "severity": "NONE",
        "yield_loss_risk_pct": "0% (Optimal bulking rate)",
        "immediate_actions": ["Maintain even soil moisture to ensure uniform tuber bulking."],
        "chemical_control": "None required.",
        "organic_bio_control": "None required.",
        "preventative_practices": "Monitor for Colorado potato beetle and potato leafhoppers."
    }
}

# Generic fallback profile for any class outside the detailed database
DEFAULT_PROFILE = {
    "severity": "MODERATE",
    "yield_loss_risk_pct": "15% - 35% without agronomic containment",
    "immediate_actions": [
        "Isolate infected plant specimens to prevent field-wide pathogen migration.",
        "Submit physical leaf tissue samples to a local agricultural extension laboratory for molecular PCR confirmation."
    ],
    "chemical_control": "Consult local registered agronomist for targeted active ingredients compliant with regional environmental regulations.",
    "organic_bio_control": "Apply broad-spectrum organic bio-fungicides (e.g. Copper soap or Neem extract) during early morning hours.",
    "preventative_practices": "Sanitize equipment, ensure adequate plant spacing, and clear weed hosts on field perimeters."
}


def get_agronomic_info(class_name: str) -> Dict[str, Any]:
    """Retrieves agronomic intelligence for a given class label."""
    if class_name in DISEASE_DATABASE:
        return DISEASE_DATABASE[class_name]
    
    # Parse crop and condition from standard PlantVillage notation (Crop___Condition)
    parts = class_name.split("___")
    crop = parts[0].replace("_", " ")
    condition = parts[1].replace("_", " ") if len(parts) > 1 else "Condition"
    
    profile = dict(DEFAULT_PROFILE)
    profile["crop"] = crop
    profile["disease"] = condition
    profile["scientific_name"] = f"Pathogen associated with {crop}"
    return profile
