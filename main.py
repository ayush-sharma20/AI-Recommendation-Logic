"""
DecodeLabs Industrial Training Kit - Project 3: AI Recommendation Logic
Engine Type: Content-Based Filtering with Weighted Token Normalization
"""

import numpy as np

# =====================================================================
# 1. THE DATASET CATALOG & SHARED VOCABULARY
# =====================================================================
# A strictly shared vocabulary space ensures similarity calculations do not fail
# due to naming discrepancies (e.g., 'Web Design' vs 'Frontend').
VOCABULARY = [
    "Python", "Java", "Algorithms", "Data Science", 
    "Web Design", "Cloud", "Automation", "Neural Networks"
]

ITEMS_CATALOG = [
    {"id": 101, "title": "Advanced Python & Data Structures", "tags": ["Python", "Algorithms"]},
    {"id": 102, "title": "Introduction to Web Design & Frontend", "tags": ["Web Design"]},
    {"id": 103, "title": "Cloud Architectures and Automation Pipelines", "tags": ["Cloud", "Automation"]},
    {"id": 104, "title": "Deep Learning & Complex Neural Networks", "tags": ["Python", "Neural Networks", "Algorithms"]},
    {"id": 105, "title": "Enterprise Java Systems & Cloud Storage", "tags": ["Java", "Cloud"]}
]

# Calculate global frequency weights to penalize generic tags and reward rare ones
GLOBAL_TAG_COUNTS = {}
for item in ITEMS_CATALOG:
    for tag in item["tags"]:
        GLOBAL_TAG_COUNTS[tag] = GLOBAL_TAG_COUNTS.get(tag, 0) + 1

# =====================================================================
# 2. CORE PROCESSING LOGIC (VECTOR MAPPING & SIMILARITY)
# =====================================================================
def transform_to_binary_vector(tags, vocabulary):
    """Transforms raw text attributes into a unified mathematical array."""
    vector = [1 if feature in tags else 0 for feature in vocabulary]
    return np.array(vector)

def calculate_jaccard_similarity(vector_a, vector_b):
    """Computes size of intersection divided by size of union."""
    intersection = np.sum(np.logical_and(vector_a, vector_b))
    union = np.sum(np.logical_or(vector_a, vector_b))
    if union == 0:
        return 0.0
    return float(intersection / union)

def compute_recommendation_score(user_tags, item_tags):
    """
    Advanced matching logic: Combines Jaccard index with contextual feature
    penalties to reward highly specific target matches over generic tags.
    """
    user_vec = transform_to_binary_vector(user_tags, VOCABULARY)
    item_vec = transform_to_binary_vector(item_tags, VOCABULARY)
    
    # Base similarity score
    base_similarity = calculate_jaccard_similarity(user_vec, item_vec)
    
    # Calculate specialized weights for overlapping elements
    overlap_bonus = 0.0
    shared_tags = set(user_tags).intersection(set(item_tags))
    
    for tag in shared_tags:
        # If a tag appears everywhere in the catalog, penalize its impact weight
        frequency = GLOBAL_TAG_COUNTS.get(tag, 1)
        tag_weight = 1.0 / frequency 
        overlap_bonus += tag_weight
        
    return base_similarity + (overlap_bonus * 0.1)

# =====================================================================
# 3. ENGINE EXECUTION PIPELINE (INPUT -> PROCESS -> OUTPUT)
# =====================================================================
def get_top_n_recommendations(user_preferences, catalog, top_n=3):
    """Processes user state inputs and outputs a truncated, high-scoring Top-N list."""
    scored_items = []
    
    print(f"--- Processing Input State: Preferences -> {user_preferences} ---")
    
    for item in catalog:
        # Processing Phase: Apply algorithmic distance logic
        score = compute_recommendation_score(user_preferences, item["tags"])
        
        scored_items.append({
            "id": item["id"],
            "title": item["title"],
            "match_score": round(score, 4)
        })
    
    # Output Phase: Generate ranked, tailored, truncated recommendations
    scored_items.sort(key=lambda x: x["match_score"], reverse=True)
    return scored_items[:top_n]

# =====================================================================
# 4. LIVE SIMULATION
# =====================================================================
if __name__ == "__main__":
    # Simulate an incoming active user profile state targeting specialized fields
    simulated_user_preferences = ["Python", "Algorithms", "Neural Networks"]
    
    recommendations = get_top_n_recommendations(
        user_preferences=simulated_user_preferences, 
        catalog=ITEMS_CATALOG, 
        top_n=3
    )
    
    print("\n--- Output Generated: Top-N Truncated List ---")
    for rank, rec in enumerate(recommendations, 1):
        print(f"Rank {rank}: [ID: {rec['id']}] {rec['title']} (Score: {rec['match_score']})")


--- Output Generated: Top-N Truncated List ---
Rank 1: [ID: 104] Deep Learning & Complex Neural Networks (Score: 1.2)
Rank 2: [ID: 101] Advanced Python & Data Structures (Score: 0.7667)
Rank 3: [ID: 102] Introduction to Web Design & Frontend (Score: 0.0)
