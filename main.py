
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


############## CANDOR Routing Components ##############

# Query complexity patterns
SIMPLE_PATTERNS = [
    r'\b(what is|what\'s|how is|how\'s|where is|when is|who is)\b',
    r'\b(get|set|send|play|create|search|find|remind|check)\b',
    r'^.{1,50}$',  # Short queries are usually simple
]

COMPLEX_PATTERNS = [
    r'\b(why|explain|describe|justify|reasoning|motivation|understand why|walk me through)\b',
    r'\band\b.*\band\b.*\band\b',  # 3+ clauses = complex
    r'\b(multiple|several|various|different)\b',
]

# Stress/uncertainty markers
FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "you know", "basically",
    "literally", "actually", "sort of", "kind of", "i mean",
    "right", "so", "well", "okay", "hmm", "just", "really"
}

UNCERTAINTY_PATTERNS = [
    r'\b(maybe|perhaps|possibly|might|could be|not sure|i think|probably)\b',
    r'\b(and|or)\b.*\b(and|or)\b',  # Multiple options = uncertainty
    r'\?',  # Questions in response = uncertainty
]

# Routing thresholds (tunable)
STRESS_THRESHOLD_LOW = 0.25   # Below this -> always local
STRESS_THRESHOLD_HIGH = 0.65  # Above this -> always cloud
CONFIDENCE_TIEBREAKER = 0.60  # In ambiguous zone, use this as fallback


def classify_query_complexity(messages):
    """
    Classify query complexity before calling local model.
    Returns: "simple", "complex", or "ambiguous"
    """
    last_msg = messages[-1]["content"].lower().strip()

    for pattern in COMPLEX_PATTERNS:
        if re.search(pattern, last_msg):
            return "complex"

    for pattern in SIMPLE_PATTERNS:
        if re.search(pattern, last_msg):
            return "simple"

    return "ambiguous"


def compute_stress_score(text, session_response_lengths):
    """
    Compute simulated stress score from text patterns.
    Returns: 0.0 (confident) to 1.0 (stressed/uncertain)

    For voice transcriptions: uses filler words, length variance, repetition
    For text queries: simulates stress via uncertainty markers and complexity
    """
    words = text.lower().split()
    if not words:
        return 0.5  # Unknown = middle ground

    # Component 1: Filler/uncertainty word density (50%)
    filler_count = sum(1 for w in words if w.rstrip('.,!?') in FILLER_WORDS)
    filler_density = min(filler_count / len(words) * 5, 1.0)

    # Add uncertainty pattern bonus
    uncertainty_bonus = 0.0
    for pattern in UNCERTAINTY_PATTERNS:
        if re.search(pattern, text.lower()):
            uncertainty_bonus += 0.15
    uncertainty_score = min(filler_density + uncertainty_bonus, 1.0)

    # Component 2: Length variance from session baseline (30%)
    current_len = len(words)
    if session_response_lengths:
        baseline = sum(session_response_lengths) / len(session_response_lengths)
        length_ratio = current_len / max(baseline, 1)
        if length_ratio < 0.5 or length_ratio > 2.0:
            length_variance = 1.0
        else:
            length_variance = abs(1.0 - length_ratio) / 1.5
    else:
        length_variance = 0.0

    # Component 3: Repetition rate (20%)
    if len(words) > 1:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        unique_bigrams = set(bigrams)
        repetition_rate = 1.0 - (len(unique_bigrams) / max(len(bigrams), 1))
    else:
        repetition_rate = 0.0

    # Weighted composite
    stress = (
        uncertainty_score * 0.50 +
        length_variance * 0.30 +
        repetition_rate * 0.20
    )
    return min(round(stress, 3), 1.0)


class ClaimRegistry:
    """
    Tracks factual claims across conversation to detect contradictions
    and identify when new territory is being explored.
    """
    def __init__(self):
        self.claims = {}  # topic_key -> claim_text
        self.topics_covered = set()

    def _extract_claims(self, text):
        """Extract simple claims from text (basic heuristic)."""
        claims = []
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            s = sentence.strip().lower()
            # Look for declarative patterns
            if len(s.split()) >= 3:
                words = s.split()
                # Use first meaningful word as topic key
                topic_key = words[0] if words[0] not in ['the', 'a', 'an'] else (words[1] if len(words) > 1 else words[0])
                claims.append((topic_key[:20], s))
        return claims

    def check_and_update(self, new_text):
        """
        Check for contradictions and new territory.
        Returns: {"contradictions": [(prev, curr), ...], "new_territory": bool}
        """
        new_claims = self._extract_claims(new_text)
        contradictions = []
        new_topics = []

        for topic_key, claim_text in new_claims:
            if topic_key in self.claims:
                # Topic seen before - check for contradiction
                if self.claims[topic_key] != claim_text:
                    contradictions.append((self.claims[topic_key], claim_text))
                self.claims[topic_key] = claim_text
            else:
                # New topic
                new_topics.append(topic_key)
                self.claims[topic_key] = claim_text
                self.topics_covered.add(topic_key)

        # New territory = introducing 2+ genuinely new topics
        new_territory = len(new_topics) >= 2

        return {
            "contradictions": contradictions,
            "new_territory": new_territory
        }


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    # Initialize the local FunctionGemma model via Cactus.
    model = cactus_init(functiongemma_path)

    # Convert generic tool schema into Cactus tool wrapper format.
    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    # Run local tool-calling completion, forcing a tool-call response.
    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    # Always free model memory after the call.
    cactus_destroy(model)

    # Parse the JSON response and return a minimal result shape for routing.
    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    # Create a Gemini client using the API key from environment.
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Convert generic tool schema into Gemini function declarations.
    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    # Gemini expects a list of user contents (no system/assistant roles here).
    contents = [m["content"] for m in messages if m["role"] == "user"]

    # Time the cloud request for benchmarking.
    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    # Extract any function calls from Gemini response candidates.
    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    CANDOR three-signal hybrid routing algorithm.
    Routes based on query complexity, stress patterns, and temporal consistency.
    """
    # Initialize session state (persists across calls via function attributes)
    if not hasattr(generate_hybrid, '_candor_registry'):
        generate_hybrid._candor_registry = ClaimRegistry()
        generate_hybrid._candor_response_lengths = []
        generate_hybrid._candor_call_count = 0
        generate_hybrid._candor_local_count = 0
        generate_hybrid._candor_cloud_count = 0

    generate_hybrid._candor_call_count += 1

    # Extract last user message
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "")
            break

    # SIGNAL 1: Pre-router (query complexity)
    complexity = classify_query_complexity(messages)

    if complexity == "complex":
        generate_hybrid._candor_cloud_count += 1
        result = generate_cloud(messages, tools)
        result["source"] = "cloud (complex query pre-routed)"
        result["_candor_reason"] = "complex query pattern detected"
        result["_candor_stress"] = None
        result["_candor_route"] = "cloud"
        return result

    # SIGNAL 2: Stress score (simulated from text patterns)
    stress = compute_stress_score(last_user_msg, generate_hybrid._candor_response_lengths)
    generate_hybrid._candor_response_lengths.append(len(last_user_msg.split()))

    # SIGNAL 3: Temporal consistency
    consistency = generate_hybrid._candor_registry.check_and_update(last_user_msg)
    contradictions = consistency["contradictions"]
    new_territory = consistency["new_territory"]

    # ROUTING DECISION TREE

    # Contradiction detected -> press locally (simple pattern matching)
    if contradictions:
        local = generate_cactus(messages, tools)
        local["source"] = "on-device"
        local["_candor_reason"] = f"contradiction detected (local sufficient)"
        local["_candor_stress"] = stress
        local["_candor_route"] = "local"

        # Check if local failed via cloud_handoff or very low confidence
        if local.get("confidence", 0) < 0.3:
            generate_hybrid._candor_cloud_count += 1
            result = generate_cloud(messages, tools)
            result["source"] = "cloud (local failed on contradiction)"
            result["total_time_ms"] += local["total_time_ms"]
            return result

        generate_hybrid._candor_local_count += 1
        return local

    # New territory + moderate/high stress -> cloud for richer reasoning
    if new_territory and stress > STRESS_THRESHOLD_LOW:
        generate_hybrid._candor_cloud_count += 1
        result = generate_cloud(messages, tools)
        result["source"] = "cloud (new territory + elevated stress)"
        result["_candor_reason"] = f"new territory with stress={stress:.2f}"
        result["_candor_stress"] = stress
        result["_candor_route"] = "cloud"
        return result

    # Low stress -> confident, use local
    if stress < STRESS_THRESHOLD_LOW:
        local = generate_cactus(messages, tools)
        local["source"] = "on-device"
        local["_candor_reason"] = f"low stress ({stress:.2f}), confident"
        local["_candor_stress"] = stress
        local["_candor_route"] = "local"

        # Fallback if local confidence very low
        if local.get("confidence", 0) < 0.3:
            generate_hybrid._candor_cloud_count += 1
            result = generate_cloud(messages, tools)
            result["source"] = "cloud (local confidence too low)"
            result["total_time_ms"] += local["total_time_ms"]
            return result

        generate_hybrid._candor_local_count += 1
        return local

    # High stress -> evasion/uncertainty, use cloud
    if stress > STRESS_THRESHOLD_HIGH:
        generate_hybrid._candor_cloud_count += 1
        result = generate_cloud(messages, tools)
        result["source"] = "cloud (high stress detected)"
        result["_candor_reason"] = f"high stress ({stress:.2f}), uncertainty detected"
        result["_candor_stress"] = stress
        result["_candor_route"] = "cloud"
        return result

    # Ambiguous zone (STRESS_THRESHOLD_LOW to STRESS_THRESHOLD_HIGH)
    # Try local, use confidence as tiebreaker
    local = generate_cactus(messages, tools)
    local["_candor_stress"] = stress

    if local.get("confidence", 0) >= CONFIDENCE_TIEBREAKER:
        local["source"] = "on-device"
        local["_candor_reason"] = f"ambiguous stress ({stress:.2f}), good confidence ({local['confidence']:.2f})"
        local["_candor_route"] = "local"
        generate_hybrid._candor_local_count += 1
        return local
    else:
        generate_hybrid._candor_cloud_count += 1
        result = generate_cloud(messages, tools)
        result["source"] = "cloud (ambiguous stress + low confidence)"
        result["_candor_reason"] = f"stress={stress:.2f}, confidence={local.get('confidence', 0):.2f}"
        result["_candor_stress"] = stress
        result["_candor_route"] = "cloud"
        result["local_confidence"] = local.get("confidence", 0)
        result["total_time_ms"] += local["total_time_ms"]
        return result


def print_result(label, result):
    """Pretty-print a generation result."""
    # Display metadata and all tool calls in a readable format.
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
