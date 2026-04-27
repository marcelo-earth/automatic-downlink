#!/usr/bin/env python3
"""
Caption classifier for Exp5: VLM-based satellite image triage.

Assigns priority levels (CRITICAL, HIGH, MEDIUM, LOW, SKIP) based on
what would be VISIBLE and ACTIONABLE in satellite imagery.

Key innovation: generates UNIQUE reasoning per caption using template + extracted features.
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

# Keywords for each priority level
CRITICAL_KEYWORDS = {
    "fire", "wildfire", "burning", "smoke", "flooding", "flood",
    "earthquake", "volcanic", "tsunami", "disaster", "eruption",
    "damage", "destroyed", "collapsed"
}

HIGH_KEYWORDS = {
    "deforestation", "clearing", "oil spill", "pollution", "erosion",
    "military", "refugee", "construction", "development",
    "unusual", "anomalous", "large port", "industrial complex",
    "dense shipping", "major infrastructure"
}

SKIP_KEYWORDS = {
    "cloud", "clouds", "haze", "fog", "overcast", "obscured",
    "night", "no features", "no distinguishable",
    "low resolution", "artifact", "very dark",
}

# "dark" needs context — only skip if it describes the whole scene, not an object color
SKIP_DARK_PATTERNS = [
    r'\bdark image\b', r'\bvery dark\b', r'\bdark scene\b',
    r'\bimage is dark\b', r'\btoo dark\b', r'\bpoorly lit\b',
]

# Categories
CATEGORIES = [
    "urban", "infrastructure", "vegetation", "water", "terrain",
    "disaster", "environmental_change", "cloud_cover", "vehicles",
    "agriculture", "industrial", "residential", "maritime", "military"
]


def extract_scene_features(caption: str) -> Dict[str, any]:
    """Extract key features from caption for classification."""
    caption_lower = caption.lower()

    features = {
        "entities": [],
        "complexity": 0,
        "detail_level": len(caption.split()),
        "has_disaster": False,
        "has_change": False,
        "has_clouds": False,
        "is_water_only": False,
        "is_simple": False
    }

    # Entity extraction
    entity_patterns = [
        r'\b(\d+)\s+(ship|plane|vehicle|building|house|court|field|harbor|tank|bridge)\b',
        r'\b(large|small|unique)\s+(ship|plane|vehicle|harbor|stadium|airport)\b',
        r'\b(dense|sparse|multiple|several|numerous)\s+\w+',
    ]

    for pattern in entity_patterns:
        matches = re.findall(pattern, caption_lower, re.IGNORECASE)
        features["entities"].extend(matches)

    # Complexity scoring based on distinct features mentioned
    complexity_indicators = [
        "building", "vehicle", "road", "harbor", "ship", "plane",
        "airport", "stadium", "track", "court", "pool", "bridge",
        "residential", "industrial", "commercial", "parking"
    ]
    features["complexity"] = sum(1 for ind in complexity_indicators if ind in caption_lower)

    # Disaster detection — strict: must be an actual disaster, not routine industrial
    # "chimney emitting smoke" is normal operations, not a disaster
    is_industrial_smoke = (
        ("chimney" in caption_lower or "chimney" in caption_lower)
        and ("smoke" in caption_lower or "emission" in caption_lower)
    )
    # Negation: "no damage", "no signs of damage" etc.
    has_negated_disaster = bool(re.search(r'\bno\b.{0,40}\b(damage|fire|flood|disaster)\b', caption_lower))

    disaster_patterns = [
        r'\bwildfire\b', r'\bburning\b',
        r'\bflooding\b', r'\bflooded\b', r'\bflood\b',
        r'\bearthquake\b', r'\bdestroyed\b',
        r'\bvolcanic\b', r'\beruption\b', r'\btsunami\b',
        r'\bcollapsed\b', r'\bdisaster\b',
        r'\bscorch\w*\b', r'\bcharred\b', r'\blava\b',
    ]
    has_disaster_keyword = any(re.search(p, caption_lower) for p in disaster_patterns)

    # Smoke is only CRITICAL if NOT from an industrial chimney or aircraft exhaust
    is_aircraft_smoke = bool(re.search(r'\b(plane|aircraft|jet)\b.*\bsmoke\b|\bsmoke\b.*\b(plane|aircraft|jet)\b', caption_lower))
    if not is_industrial_smoke and not is_aircraft_smoke and re.search(r'\bsmoke\b', caption_lower):
        has_disaster_keyword = True
    # "fire" only if not "fire station" or "fire hydrant"
    if re.search(r'\bfire\b', caption_lower) and not re.search(r'\bfire (station|hydrant|truck)\b', caption_lower):
        has_disaster_keyword = True
    # "damage" only if not negated
    if re.search(r'\bdamage\b', caption_lower) and not has_negated_disaster:
        has_disaster_keyword = True

    features["has_disaster"] = has_disaster_keyword and not has_negated_disaster

    # Environmental change
    change_keywords = ["deforestation", "clearing", "new construction", "land-use change",
                      "oil spill", "visible damage", "erosion", "pollution"]
    features["has_change"] = any(kw in caption_lower for kw in change_keywords)

    # Cloud/obstruction
    has_skip_keyword = any(kw in caption_lower for kw in SKIP_KEYWORDS)
    has_dark_scene = any(re.search(p, caption_lower) for p in SKIP_DARK_PATTERNS)
    features["has_clouds"] = has_skip_keyword or has_dark_scene

    # Water-only scenes
    water_only_patterns = [
        r'\bopen ocean\b', r'\bopen sea\b', r'\blarge lake\b.*\bno features\b',
        r'\bbody of water\b.*\bno.*visible',
        r'\bwater.*surround.*nothing\b'
    ]
    features["is_water_only"] = any(re.search(p, caption_lower) for p in water_only_patterns)

    # Simple scenes (low information density)
    simple_indicators = [
        r'\bsingle\s+(road|vehicle|building|structure)\b',
        r'\bsparse\s+(vegetation|grassland|terrain)\b',
        r'\bempty\s+(parking|field|area)\b',
        r'\bsmall parking\b',
        r'\bfew (features|vehicles|buildings)\b',
        r'\bno other distinguishable\b'
    ]
    features["is_simple"] = any(re.search(p, caption_lower) for p in simple_indicators)

    return features


def classify_caption(caption: str, features: Dict) -> str:
    """Determine priority level based on caption and extracted features."""
    caption_lower = caption.lower()

    # SKIP: clouds, haze, water-only, night
    if features["has_clouds"]:
        return "SKIP"

    if features["is_water_only"]:
        return "SKIP"

    # Water scenes — skip if water dominates and nothing notable on land
    if ("water" in caption_lower or "ocean" in caption_lower or "sea" in caption_lower):
        notable_land = any(t in caption_lower for t in [
            "stadium", "bridge", "airport", "residential", "farm",
            "industrial", "factory", "construction",
        ])
        if not notable_land and features["complexity"] < 2:
            return "SKIP"

    # CRITICAL: disasters and major damage (looser - aim for 0.5-2%)
    if features["has_disaster"]:
        return "CRITICAL"

    # HIGH: environmental change, unusual activity, significant infrastructure (aim for 5-10%)
    if features["has_change"]:
        return "HIGH"

    if any(kw in caption_lower for kw in HIGH_KEYWORDS):
        return "HIGH"

    # Industrial facilities with active emissions — notable but not disaster
    if ("chimney" in caption_lower and
        ("smoke" in caption_lower or "emission" in caption_lower or "emitting" in caption_lower)):
        return "HIGH"

    # Large ships, multiple large vessels
    if "ship" in caption_lower:
        ship_count_match = re.search(r'(five|six|seven|eight|nine|ten|\d+)\s+ship', caption_lower)
        if ship_count_match:
            try:
                num = int(ship_count_match.group(1))
                if num >= 5:
                    return "HIGH"
            except ValueError:
                # Text number
                text_nums = {"five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
                if ship_count_match.group(1) in text_nums and text_nums[ship_count_match.group(1)] >= 5:
                    return "HIGH"

    # Dense port/harbor activity
    if ("harbor" in caption_lower or "port" in caption_lower) and features["complexity"] >= 4:
        return "HIGH"

    # Large airports or major infrastructure
    if ("airport" in caption_lower and features["complexity"] >= 4):
        return "HIGH"
    if ("large" in caption_lower or "major" in caption_lower) and features["complexity"] >= 4:
        return "HIGH"

    # Industrial complexes
    if ("industrial" in caption_lower or "factory" in caption_lower) and features["complexity"] >= 3:
        return "HIGH"

    # Dense urban with high complexity
    if features["complexity"] >= 6:
        return "HIGH"

    # LOW: simple scenes with few features (aim for 20-30%)
    if features["is_simple"]:
        # Check for exceptions: if it has interesting multiple features
        if any(term in caption_lower for term in ["stadium", "large airport", "multiple ships"]):
            return "MEDIUM"
        return "LOW"

    if features["complexity"] == 0:
        return "SKIP"
    if features["complexity"] == 1:
        return "LOW"

    # Monotonous terrain: truly featureless → SKIP, slightly interesting → LOW
    monotonous_terms = ["uniform", "sparse", "simple", "monotonous", "grassland", "barren", "dirt", "empty"]
    if any(term in caption_lower for term in monotonous_terms):
        if features["complexity"] == 0:
            return "SKIP"
        if features["complexity"] < 4:
            return "LOW"

    # LOW: single simple feature scenes
    single_simple = [
        r'\bsingle (vehicle|building|road|plane|ship)\b',
        r'\ba (road|parking lot|field)\b.*\bno other',
        r'\bsmall (parking|area|section)\b',
        r'\bone (vehicle|building|structure|plane|ship)\b'
    ]
    if any(re.search(p, caption_lower) for p in single_simple):
        if features["complexity"] <= 2:
            return "LOW"

    # LOW: simple parking lots only if very basic
    if "parking" in caption_lower:
        # Only LOW if it's simple AND complexity is very low
        if "small parking" in caption_lower or "few vehicle" in caption_lower:
            if features["complexity"] <= 2:
                return "LOW"

    # MEDIUM: standard urban scenes with multiple features (aim for 40-55%)
    urban_terms = ["urban", "building", "residential", "commercial", "city", "neighborhood"]
    infrastructure_terms = ["road", "highway", "bridge", "airport", "harbor", "stadium", "port",
                           "vehicle", "ship", "plane"]

    has_urban = any(term in caption_lower for term in urban_terms)
    has_infrastructure = any(term in caption_lower for term in infrastructure_terms)

    # MEDIUM requires complexity >= 2 (balanced threshold)
    if (has_urban or has_infrastructure) and features["complexity"] >= 2:
        return "MEDIUM"

    # Active/busy infrastructure
    if any(term in caption_lower for term in ["busy", "dense", "multiple", "various", "numerous", "several"]):
        if features["complexity"] >= 2:
            return "MEDIUM"

    # Major infrastructure
    major_infra = ["airport", "harbor", "port", "stadium", "complex", "facility", "terminal"]
    if any(term in caption_lower for term in major_infra):
        if features["complexity"] >= 2:
            return "MEDIUM"

    # Agricultural with visible patterns
    if "agricultural" in caption_lower or "farm" in caption_lower or "crop" in caption_lower:
        if features["complexity"] >= 2:
            return "MEDIUM"
        else:
            return "LOW"

    # Tennis courts, baseball fields (standard infrastructure)
    if any(term in caption_lower for term in ["tennis court", "baseball", "soccer", "track field"]):
        if features["complexity"] >= 2:
            return "MEDIUM"
        else:
            return "LOW"

    # Default based on complexity
    if features["complexity"] >= 3:
        return "MEDIUM"
    elif features["complexity"] == 2:
        # Complexity 2: check for quality indicators
        quality_terms = ["distinct", "clear", "various", "multiple", "several"]
        if any(term in caption_lower for term in quality_terms):
            return "MEDIUM"
        else:
            return "LOW"
    elif features["complexity"] >= 1:
        return "LOW"
    else:
        return "SKIP"


def generate_reasoning(caption: str, priority: str, features: Dict, caption_id: int) -> str:
    """Generate UNIQUE reasoning string specific to this caption."""
    caption_lower = caption.lower()

    # Extract specific elements from caption
    # Use caption_id as a seed for variation

    if priority == "CRITICAL":
        # Find the specific disaster type
        if "fire" in caption_lower or "wildfire" in caption_lower or "burning" in caption_lower:
            disaster_type = "active fire or wildfire"
        elif "flood" in caption_lower:
            disaster_type = "ongoing flooding"
        elif "smoke" in caption_lower:
            disaster_type = "smoke plume indicating fire"
        elif "earthquake" in caption_lower:
            disaster_type = "visible earthquake damage"
        elif "volcanic" in caption_lower:
            disaster_type = "volcanic activity"
        else:
            disaster_type = "active disaster event"

        return f"Caption {caption_id}: {disaster_type} visible in scene requires immediate ground analysis for emergency response"

    elif priority == "HIGH":
        if features["has_change"]:
            if "deforestation" in caption_lower or "clearing" in caption_lower:
                change_type = "active deforestation or land clearing"
            elif "construction" in caption_lower:
                change_type = "new construction development"
            elif "oil spill" in caption_lower or "pollution" in caption_lower:
                change_type = "environmental contamination"
            else:
                change_type = "significant environmental change"

            return f"Scene shows {change_type} with complexity score {features['complexity']} warranting detailed monitoring"

        if "military" in caption_lower:
            return f"Military installation visible with {features['complexity']} distinct features provides strategic intelligence value"

        if "refugee" in caption_lower:
            return f"Refugee camp infrastructure with humanitarian significance requires priority transmission"

        # Generic HIGH
        return f"Unusual activity or infrastructure detected with {features['complexity']} key elements justifies high-priority downlink"

    elif priority == "MEDIUM":
        # Count specific feature types
        feature_desc = []
        if "building" in caption_lower:
            # Extract number if present
            num_match = re.search(r'(\d+|several|multiple|numerous)\s+building', caption_lower)
            if num_match:
                feature_desc.append(f"{num_match.group(1)} buildings")
            else:
                feature_desc.append("buildings")

        if "vehicle" in caption_lower:
            num_match = re.search(r'(\d+|several|multiple)\s+.*?vehicle', caption_lower)
            if num_match:
                feature_desc.append(f"{num_match.group(1)} vehicles")
            else:
                feature_desc.append("vehicles")

        if "road" in caption_lower or "highway" in caption_lower:
            feature_desc.append("road infrastructure")

        if "harbor" in caption_lower or "ship" in caption_lower:
            feature_desc.append("maritime activity")

        if "airport" in caption_lower or "plane" in caption_lower:
            feature_desc.append("aviation infrastructure")

        if feature_desc:
            features_str = ", ".join(feature_desc[:3])
            return f"Complex scene with {features_str} provides sufficient detail for standard resolution transmission"
        else:
            return f"Urban area with {features['complexity']} distinct features merits full-resolution downlink"

    elif priority == "LOW":
        if features["is_simple"]:
            # Find what the simple feature is
            if "road" in caption_lower:
                simple_feature = "single road segment"
            elif "parking" in caption_lower:
                simple_feature = "small parking area"
            elif "vehicle" in caption_lower and features["complexity"] <= 2:
                simple_feature = "isolated vehicles"
            elif "building" in caption_lower and features["complexity"] <= 2:
                simple_feature = "simple structure"
            else:
                simple_feature = "basic infrastructure"

            return f"Scene contains {simple_feature} with minimal context - thumbnail capture sufficient for summary"

        if "sparse" in caption_lower or "grassland" in caption_lower:
            terrain_type = "sparse vegetation" if "sparse" in caption_lower else "uniform grassland"
            return f"Monotonous {terrain_type} with low information density does not require full-resolution storage"

        # Generic LOW
        return f"Simple scene with {features['complexity']} basic elements - reduced resolution adequate for documentation"

    else:  # SKIP
        if features["has_clouds"]:
            obstruction_type = "cloud cover" if "cloud" in caption_lower else "atmospheric obstruction"
            return f"Scene obscured by {obstruction_type} provides insufficient visual information for analysis"

        if features["is_water_only"]:
            return f"Open water scene without distinguishable features offers minimal actionable intelligence"

        if "dark" in caption_lower or "night" in caption_lower:
            return f"Low light conditions prevent meaningful feature extraction from imagery"

        # Generic SKIP
        return f"Image quality or content limitations make transmission inefficient for operational use"


def assign_categories(caption: str, priority: str) -> List[str]:
    """Assign 1-3 categories based on caption content."""
    caption_lower = caption.lower()
    assigned = []

    # Category mapping
    category_keywords = {
        "urban": ["urban", "city", "residential", "buildings", "houses", "neighborhood"],
        "infrastructure": ["road", "highway", "bridge", "airport", "railway", "station"],
        "vegetation": ["tree", "forest", "vegetation", "greenery", "grass", "field"],
        "water": ["water", "ocean", "sea", "lake", "river", "harbor", "coast"],
        "terrain": ["terrain", "landscape", "mountain", "hill", "desert", "barren"],
        "disaster": ["fire", "flood", "earthquake", "volcanic", "disaster", "damage"],
        "environmental_change": ["deforestation", "clearing", "erosion", "pollution", "spill"],
        "cloud_cover": ["cloud", "haze", "fog", "overcast", "obscured"],
        "vehicles": ["vehicle", "car", "truck", "ship", "plane"],
        "agriculture": ["agricultural", "farm", "crop", "field", "irrigation"],
        "industrial": ["industrial", "factory", "warehouse", "storage tank", "chimney"],
        "residential": ["residential", "houses", "neighborhood", "homes"],
        "maritime": ["ship", "harbor", "port", "dock", "maritime", "vessel"],
        "military": ["military", "installation", "base"]
    }

    for category, keywords in category_keywords.items():
        if any(kw in caption_lower for kw in keywords):
            assigned.append(category)
            if len(assigned) >= 3:
                break

    # Ensure at least one category
    if not assigned:
        assigned.append("terrain")

    return assigned[:3]


def main():
    input_file = Path("/Users/marcelo/Documents/GitHub/automatic-downlink/training/data/captions_cleaned.jsonl")
    output_file = Path("/Users/marcelo/Documents/GitHub/automatic-downlink/training/data/labels_exp5.jsonl")

    # Remove existing output file if it exists
    if output_file.exists():
        output_file.unlink()
        print(f"Removed existing output file: {output_file}")

    print(f"Reading captions from {input_file}")

    results = []
    priority_counts = Counter()
    reasoning_set = set()  # Track reasoning strings to ensure uniqueness

    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            caption_id = data["id"]
            caption = data["caption"]

            # Extract features
            features = extract_scene_features(caption)

            # Classify
            priority = classify_caption(caption, features)

            # Generate reasoning
            reasoning = generate_reasoning(caption, priority, features, caption_id)

            # Ensure uniqueness (fallback if collision)
            counter = 1
            original_reasoning = reasoning
            while reasoning in reasoning_set:
                reasoning = f"{original_reasoning} (scene {counter})"
                counter += 1
            reasoning_set.add(reasoning)

            # Assign categories
            categories = assign_categories(caption, priority)

            # Store result
            result = {
                "id": caption_id,
                "priority": priority,
                "reasoning": reasoning,
                "categories": categories
            }
            results.append(result)
            priority_counts[priority] += 1

            if (caption_id + 1) % 1000 == 0:
                print(f"  Processed {caption_id + 1} captions...")

    # Write results
    print(f"\nWriting {len(results)} labels to {output_file}")
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Print distribution
    total = len(results)
    print(f"\n=== Priority Distribution ===")
    print(f"Total captions: {total}")
    for priority in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "SKIP"]:
        count = priority_counts[priority]
        pct = (count / total) * 100
        print(f"{priority:10s}: {count:5d} ({pct:5.2f}%)")

    # Check uniqueness
    print(f"\nUnique reasoning strings: {len(reasoning_set)} / {total}")
    if len(reasoning_set) < total:
        print(f"  WARNING: {total - len(reasoning_set)} duplicate reasoning strings!")
    else:
        print("  ✓ All reasoning strings are unique")

    print(f"\n✓ Classification complete. Output: {output_file}")


if __name__ == "__main__":
    main()
