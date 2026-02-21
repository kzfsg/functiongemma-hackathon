# CANDOR Implementation

**CANDOR** - A voice-driven adaptive questioning agent that routes to cloud based on **human uncertainty**, not just model uncertainty.

## What We Implemented

### Core Concept
Routes to cloud when the *human* is uncertain, detected from:
1. **Query Complexity Pre-Router**: Classifies queries BEFORE attempting local inference
2. **Stress Pattern Detection**: Simulates stress from linguistic uncertainty markers in text
3. **Temporal Consistency**: Tracks claims to detect contradictions across conversation

### Files Modified/Created

1. **main.py** - Modified with CANDOR three-signal routing
   - Added helper functions: `classify_query_complexity()`, `compute_stress_score()`, `ClaimRegistry` class
   - Replaced `generate_hybrid()` with sophisticated routing logic
   - Function signature unchanged for benchmark compatibility

2. **candor_tools.py** - New file with specialized CANDOR tools
   - `generate_question`: Simple factual follow-ups (local-friendly)
   - `flag_inconsistency`: Catch contradictions (local-friendly)
   - `assess_credibility`: Deep analysis (cloud-triggering)

3. **demo_voice.py** - Minimal Python voice demo
   - Uses `cactus_transcribe` for voice input
   - Shows routing decisions in real-time
   - Displays session statistics

4. **record_audio.py** - Optional recording utility
   - Simple microphone recording for creating test samples
   - Requires: `pip install sounddevice scipy`

## How It Works

### Routing Decision Tree

```
User Input
    ↓
SIGNAL 1: Query Complexity
    ├─→ "complex" → CLOUD (skip local)
    ├─→ "simple" → continue
    └─→ "ambiguous" → continue
    ↓
SIGNAL 2: Stress Score (from text patterns)
    ├─→ stress < 0.25 → LOCAL (confident)
    ├─→ 0.25 ≤ stress ≤ 0.65 → Try LOCAL, use confidence as tiebreaker
    └─→ stress > 0.65 → CLOUD (uncertainty detected)
    ↓
SIGNAL 3: Temporal Consistency
    ├─→ Contradiction detected → LOCAL (simple pattern matching)
    └─→ New territory (2+ new topics) + elevated stress → CLOUD
```

### Stress Score Components

1. **Filler/Uncertainty Word Density (50%)**:
   - Filler words: um, uh, er, ah, like, you know, basically, etc.
   - Uncertainty patterns: maybe, perhaps, possibly, not sure, I think, etc.

2. **Length Variance from Session Baseline (30%)**:
   - Very short responses = terse/evasive
   - Very long responses = over-explaining
   - Both signal stress

3. **Repetition Rate (20%)**:
   - Bigram repetition indicates stalling

## Usage

### Running Benchmark
```bash
python benchmark.py
```

This tests CANDOR routing with the existing 7 generic tools (weather, alarm, message, etc.)

### Running Voice Demo
```bash
# Record audio sample
python record_audio.py 5 test.wav

# Run CANDOR demo
python demo_voice.py test.wav

# Or process multiple audio files
python demo_voice.py sample1.wav sample2.wav sample3.wav
```

## Tunable Parameters

In `main.py`:

```python
STRESS_THRESHOLD_LOW = 0.25   # Below this -> always local
STRESS_THRESHOLD_HIGH = 0.65  # Above this -> always cloud
CONFIDENCE_TIEBREAKER = 0.60  # In ambiguous zone, use this as fallback
```

### Tuning Strategy

- **If F1 score is low**: Increase cloud routing (lower `CONFIDENCE_TIEBREAKER`)
- **If on-device ratio too low**: Increase local routing (lower `STRESS_THRESHOLD_LOW`)
- **If time is high**: Aggressive local routing (increase thresholds)

## Target Metrics

- **F1 Score**: > 0.85 (tool call correctness)
- **On-device Ratio**: > 60% (edge computing priority)
- **Avg Time**: < 300ms per call
- **Total Score**: Top 10 for subjective judging

## Key Design Decisions

### Why Simulate Stress for Text?
The benchmark uses text-only messages, but CANDOR's insight about human uncertainty still applies. We detect uncertainty through:
- Linguistic markers (maybe, perhaps, not sure)
- Multiple options/clauses (or, and patterns)
- Length variance (very short = terse, very long = over-explaining)

### Why Keep Existing Tools?
Benchmark compatibility. The scoring system expects the 7 generic tools. CANDOR tools are for the qualitative demo only.

### Session State
Uses function attributes to persist state across calls:
- `generate_hybrid._candor_registry`: ClaimRegistry instance
- `generate_hybrid._candor_response_lengths`: Length history for variance calculation
- `generate_hybrid._candor_call_count`: Total calls
- `generate_hybrid._candor_local_count`: Local routing count
- `generate_hybrid._candor_cloud_count`: Cloud routing count

## Debugging

The routing result includes diagnostic fields:
- `_candor_route`: "local" or "cloud"
- `_candor_reason`: Human-readable routing explanation
- `_candor_stress`: Computed stress score (0.0-1.0)

Example:
```python
result = generate_hybrid(messages, tools)
print(f"Route: {result['_candor_route']}")
print(f"Reason: {result['_candor_reason']}")
print(f"Stress: {result['_candor_stress']}")
```

## Next Steps

1. Run `python benchmark.py` to establish baseline performance
2. Tune thresholds based on F1, on-device ratio, and timing results
3. Test voice demo with various audio samples (low stress, high stress, contradictions)
4. Submit to leaderboard: `python submit.py --team "CANDOR" --location "YourCity"`

## One-Line Pitch

**"CANDOR routes to cloud not when the model is uncertain — but when the human is, detected live from their voice."**
