# CANDOR — Implementation Plan for Cactus x Google DeepMind Hackathon

> **Pass this entire document to your IDE model before writing any code.**
> This is a self-contained build plan for CANDOR, a voice-driven truth-seeking agent
> built on FunctionGemma (on-device via Cactus) + Gemini Flash (cloud fallback).
> The hackathon repo is: https://github.com/cactus-compute/functiongemma-hackathon

---

## 1. Project Overview

**CANDOR** is an AI-powered adaptive questioning agent. It poses probing questions to a
user, listens to their spoken responses via `cactus_transcribe`, and uses three behavioral
signals extracted from their voice to decide — in real time — whether to handle the next
question locally (FunctionGemma on-device) or escalate to Gemini Flash (cloud).

The routing algorithm is the product. The Android app is the demo vehicle.

**Core insight:** Route to cloud not when the *model* is uncertain, but when the *human*
is — detected live from acoustic stress markers in their voice.

---

## 2. Repository Structure (existing)

```
functiongemma-hackathon/
├── main.py          ← PRIMARY FILE TO MODIFY (generate_hybrid method)
├── benchmark.py     ← DO NOT MODIFY (used for objective leaderboard scoring)
├── submit.py        ← DO NOT MODIFY (leaderboard submission)
├── assets/
└── README.md
```

**Your only obligation to the leaderboard:** modify the internal logic of `generate_hybrid`
in `main.py` without changing its input/output signature.

**Your submission deliverables:**
1. Modified `main.py` with CANDOR routing logic
2. Android app (APK or source) demonstrating CANDOR end-to-end
3. Short video (2 min max) showing the routing dashboard in action

---

## 3. Judging Rubrics (from README)

| Rubric | Weight | What CANDOR targets |
|--------|--------|---------------------|
| **Rubric 1** | High | Quality and cleverness of hybrid routing algorithm |
| **Rubric 2** | High | End-to-end product executing function calls for real-world problems |
| **Rubric 3** | High | Voice-to-action product leveraging `cactus_transcribe` |

CANDOR hits all three simultaneously:
- Rubric 1: Three-signal routing (complexity pre-router + voice stress + temporal consistency)
- Rubric 2: Real function calls — `generate_question`, `flag_inconsistency`, `assess_credibility`
- Rubric 3: `cactus_transcribe` is the **primary input sensor**, not an add-on

---

## 4. The CANDOR Routing Algorithm

### 4.1 Three-Signal Architecture

The routing decision is made by stacking three signals in sequence before and after
attempting local inference. Do NOT rely on the built-in `confidence_threshold` alone —
the README explicitly states you must design custom strategies.

```
User speaks
     │
     ▼
cactus_transcribe (local, always)
     │
     ▼
┌─────────────────────────────────┐
│  SIGNAL 1: Query Complexity     │  ← Pre-router, runs BEFORE local model
│  classify_query_complexity()    │
│  Output: simple | complex |     │
│          ambiguous              │
└─────────────────────────────────┘
     │
     ├── "complex" ──────────────────────────────► CLOUD (skip local entirely)
     │
     ├── "simple" ──────► SIGNAL 3 check ──► LOCAL
     │
     └── "ambiguous" ──►
                    ┌─────────────────────────────┐
                    │  SIGNAL 2: Voice Stress      │
                    │  compute_stress_score()      │
                    │  Output: 0.0 – 1.0           │
                    └─────────────────────────────┘
                         │
                         ├── stress < 0.30 ──────► LOCAL
                         ├── stress 0.30–0.65 ──► Try local, confidence tiebreaker
                         └── stress > 0.65 ──────► CLOUD

At any point where LOCAL is chosen:
     │
     ▼
┌─────────────────────────────────┐
│  SIGNAL 3: Temporal Consistency │  ← Runs on every local decision
│  ClaimRegistry.check_update()   │
│  Output: contradictions list    │
└─────────────────────────────────┘
     │
     ├── contradiction found ──► Stay LOCAL (contradiction pressing is simple)
     └── new territory ─────────► Escalate to CLOUD
```

### 4.2 Signal 1 — Query Complexity Pre-Router

Classify the incoming message complexity **before** calling FunctionGemma.
This avoids wasting prefill time on queries the 270M model cannot handle.

```python
SIMPLE_PATTERNS = [
    r'\b(what is your name|how old|where were you|when did|yes or no|did you)\b',
    r'^.{1,40}$',   # Very short queries are almost always simple tool calls
]

COMPLEX_PATTERNS = [
    r'\b(why did you|explain your reasoning|what were you thinking|'
    r'how do you justify|describe your motivation|walk me through|'
    r'what was going through your mind|help me understand why)\b',
]

def classify_query_complexity(messages: list) -> str:
    last_msg = messages[-1]["content"].lower().strip()
    for pattern in COMPLEX_PATTERNS:
        if re.search(pattern, last_msg):
            return "complex"
    for pattern in SIMPLE_PATTERNS:
        if re.search(pattern, last_msg):
            return "simple"
    return "ambiguous"
```

### 4.3 Signal 2 — Voice Stress Scorer

Extracts three acoustic proxy features from the **transcribed text** returned by
`cactus_transcribe`. Since we only have text (not raw audio waveforms), we use
linguistic markers that correlate strongly with vocal stress:

```python
FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "you know", "basically",
    "literally", "actually", "sort of", "kind of", "i mean",
    "right", "so", "well", "okay", "hmm"
}

def compute_stress_score(
    transcribed_text: str,
    session_response_lengths: list
) -> float:
    """
    Returns a stress score from 0.0 (calm/confident) to 1.0 (stressed/evasive).
    
    Three components:
    1. Filler word density (weight: 0.50)
       High fillers = hedging = uncertainty
    2. Response length variance vs session baseline (weight: 0.30)
       Sudden brevity = evasion; sudden verbosity = over-explaining
    3. Repetition rate (weight: 0.20)
       Repeating phrases = stalling tactic
    """
    words = transcribed_text.lower().split()
    if not words:
        return 0.5  # Unknown state — middle ground, use confidence tiebreaker

    # Component 1: Filler density
    filler_count = sum(1 for w in words if w.rstrip('.,!?') in FILLER_WORDS)
    filler_density = min(filler_count / len(words) * 5, 1.0)  # Scale: 20% fillers = max stress

    # Component 2: Length variance from session baseline
    current_len = len(words)
    if session_response_lengths:
        baseline = sum(session_response_lengths) / len(session_response_lengths)
        length_ratio = current_len / max(baseline, 1)
        # Both too short (< 0.4x) and too long (> 2.5x) signal stress
        if length_ratio < 0.4 or length_ratio > 2.5:
            length_variance = 1.0
        else:
            length_variance = abs(1.0 - length_ratio) / 1.5
    else:
        length_variance = 0.0  # No baseline yet — first response

    # Component 3: Repetition rate (bigram repetition)
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    unique_bigrams = set(bigrams)
    repetition_rate = 1.0 - (len(unique_bigrams) / max(len(bigrams), 1))

    # Weighted composite
    stress = (
        filler_density   * 0.50 +
        length_variance  * 0.30 +
        repetition_rate  * 0.20
    )
    return min(round(stress, 3), 1.0)
```

### 4.4 Signal 3 — Temporal Consistency Checker

Tracks factual claims made across the session. Contradiction detection is a local task
(simple pattern matching); novel territory detection escalates to cloud.

```python
class ClaimRegistry:
    """
    Session-scoped memory of factual claims.
    Detects contradictions and flags unexplored territory.
    """

    def __init__(self):
        self.claims: dict[str, str] = {}  # topic_key → last stated claim
        self.topics_covered: set[str] = set()

    def _extract_claims(self, text: str) -> list[tuple[str, str]]:
        """
        Extracts (topic_key, claim_text) pairs from transcribed response.
        Simple heuristic: first-person declarative sentences.
        """
        claims = []
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            s = sentence.strip().lower()
            if s.startswith("i ") and len(s.split()) >= 4:
                # Use the verb as the topic key
                words = s.split()
                topic_key = words[1] if len(words) > 1 else s[:20]
                claims.append((topic_key, s))
        return claims

    def check_and_update(self, new_text: str) -> dict:
        """
        Returns:
            {
                "contradictions": [(previous, current), ...],
                "new_territory": bool  # True if introducing genuinely new topics
            }
        """
        new_claims = self._extract_claims(new_text)
        contradictions = []
        new_topics = []

        for topic_key, claim_text in new_claims:
            if topic_key in self.claims:
                # Topic seen before — check for contradiction
                if self.claims[topic_key] != claim_text:
                    contradictions.append((self.claims[topic_key], claim_text))
                self.claims[topic_key] = claim_text
            else:
                # New topic introduced
                new_topics.append(topic_key)
                self.claims[topic_key] = claim_text
                self.topics_covered.add(topic_key)

        new_territory = len(new_topics) > 2  # Arbitrary threshold: 3+ new topics = complex

        return {
            "contradictions": contradictions,
            "new_territory": new_territory
        }
```

---

## 5. Full `generate_hybrid` Implementation

This is the core deliverable for the leaderboard. Paste this into `main.py` as the
internal body of `generate_hybrid`. **Do not change the method signature.**

```python
import re
import json
from google import genai

# ── Paste ClaimRegistry class here (Section 4.4) ──
# ── Paste compute_stress_score here (Section 4.3) ──
# ── Paste classify_query_complexity here (Section 4.2) ──

STRESS_THRESHOLD_LOW  = 0.30   # Below this → always local
STRESS_THRESHOLD_HIGH = 0.65   # Above this → always cloud
CONFIDENCE_TIEBREAKER = 0.55   # In ambiguous zone, use this as fallback


def generate_hybrid(self, messages, tools, **kwargs):
    """
    CANDOR three-signal hybrid routing algorithm.
    
    Routing priority:
    1. Pre-router: complex queries skip local entirely
    2. Voice stress score: high stress → cloud, low → local
    3. Temporal consistency: contradictions press locally, new territory → cloud
    4. Local attempt with confidence tiebreaker for ambiguous cases
    5. Model-triggered cloud_handoff as final safety net
    """

    # ── Session state initialisation (persists across calls) ──
    if not hasattr(self, '_candor_claim_registry'):
        self._candor_claim_registry = ClaimRegistry()
        self._candor_response_lengths = []
        self._candor_call_count = 0
        self._candor_local_count = 0
        self._candor_cloud_count = 0

    self._candor_call_count += 1

    # ── Extract last user message ──
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "")
            break

    # ── Signal 1: Pre-router (complexity classification) ──
    complexity = classify_query_complexity(messages)

    if complexity == "complex":
        self._candor_cloud_count += 1
        return self._candor_call_cloud(
            messages, tools,
            reason="pre-routed: psychologically complex query",
            stress=None,
            **kwargs
        )

    # ── Signal 2: Voice stress score ──
    stress = compute_stress_score(last_user_msg, self._candor_response_lengths)
    self._candor_response_lengths.append(len(last_user_msg.split()))

    # ── Signal 3: Temporal consistency ──
    consistency = self._candor_claim_registry.check_and_update(last_user_msg)
    contradictions = consistency["contradictions"]
    new_territory  = consistency["new_territory"]

    # ── Routing decision tree ──

    # Contradiction detected → always press locally (simple pattern matching)
    if contradictions:
        result = self._candor_call_local(messages, tools, **kwargs)
        result["_candor_reason"]  = f"contradiction detected: {contradictions[0]}"
        result["_candor_stress"]  = stress
        result["_candor_route"]   = "local"
        if result.get("cloud_handoff"):
            # Local couldn't handle even this — escalate
            self._candor_cloud_count += 1
            return self._candor_call_cloud(
                messages, tools,
                reason="local fallback after contradiction attempt",
                stress=stress,
                **kwargs
            )
        self._candor_local_count += 1
        return result

    # New territory introduced → escalate for richer reasoning
    if new_territory and stress > STRESS_THRESHOLD_LOW:
        self._candor_cloud_count += 1
        return self._candor_call_cloud(
            messages, tools,
            reason="new territory + moderate stress: complex reasoning needed",
            stress=stress,
            **kwargs
        )

    # Low stress → confident response, always local
    if stress < STRESS_THRESHOLD_LOW:
        result = self._candor_call_local(messages, tools, **kwargs)
        result["_candor_reason"] = f"low stress ({stress:.2f}): confident, local sufficient"
        result["_candor_stress"] = stress
        result["_candor_route"]  = "local"
        if result.get("cloud_handoff"):
            self._candor_cloud_count += 1
            return self._candor_call_cloud(
                messages, tools,
                reason="model-triggered handoff despite low stress",
                stress=stress,
                **kwargs
            )
        self._candor_local_count += 1
        return result

    # High stress → evasion detected, cloud for sophisticated cross-examination
    if stress > STRESS_THRESHOLD_HIGH:
        self._candor_cloud_count += 1
        return self._candor_call_cloud(
            messages, tools,
            reason=f"high stress ({stress:.2f}): evasion likely, cloud needed",
            stress=stress,
            **kwargs
        )

    # Ambiguous zone (0.30–0.65) → try local, use confidence as tiebreaker
    result = self._candor_call_local(messages, tools, **kwargs)
    confidence = result.get("confidence", 0.0)
    result["_candor_stress"] = stress

    if result.get("cloud_handoff") or confidence < CONFIDENCE_TIEBREAKER:
        self._candor_cloud_count += 1
        return self._candor_call_cloud(
            messages, tools,
            reason=f"ambiguous stress ({stress:.2f}) + low confidence ({confidence:.2f})",
            stress=stress,
            **kwargs
        )

    result["_candor_reason"] = f"ambiguous stress ({stress:.2f}), confidence ok ({confidence:.2f})"
    result["_candor_route"]  = "local"
    self._candor_local_count += 1
    return result


def _candor_call_local(self, messages, tools, **kwargs):
    """Wrapper for local FunctionGemma call."""
    raw = cactus_complete(self.model, messages, tools=tools, **kwargs)
    return json.loads(raw)


def _candor_call_cloud(self, messages, tools, reason="", stress=None, **kwargs):
    """
    Wrapper for Gemini Flash cloud call.
    Converts tools from Cactus format to Gemini function declarations.
    """
    client = genai.Client()

    # Convert tools to Gemini format
    gemini_tools = []
    if tools:
        function_declarations = []
        for tool in tools:
            function_declarations.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {})
            })
        gemini_tools = [{"function_declarations": function_declarations}]

    # Convert messages to Gemini format
    gemini_messages = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        gemini_messages.append({"role": role, "parts": [{"text": m["content"]}]})

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=gemini_messages,
            tools=gemini_tools if gemini_tools else None,
        )

        # Extract function call if present
        function_calls = []
        response_text = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                function_calls.append({
                    "name": fc.name,
                    "arguments": dict(fc.args)
                })
            elif hasattr(part, 'text') and part.text:
                response_text = part.text

        return {
            "success": True,
            "error": None,
            "cloud_handoff": False,
            "response": response_text,
            "function_calls": function_calls,
            "confidence": 1.0,  # Cloud = maximum confidence
            "_candor_reason": reason,
            "_candor_stress": stress,
            "_candor_route": "cloud",
            "time_to_first_token_ms": 0,
            "total_time_ms": 0,
            "prefill_tps": 0,
            "decode_tps": 0,
            "ram_usage_mb": 0,
            "prefill_tokens": 0,
            "decode_tokens": 0,
            "total_tokens": 0,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "cloud_handoff": False,
            "response": None,
            "function_calls": [],
            "confidence": 0.0,
            "_candor_reason": f"cloud error: {str(e)}",
            "_candor_route": "cloud_error",
        }
```

---

## 6. Tool Definitions for CANDOR

These three tools are what FunctionGemma selects from during a CANDOR session.
The descriptions are engineered to guide routing behavior — notice how each description
hints at when it should be used locally vs. via cloud.

```python
CANDOR_TOOLS = [
    {
        "name": "generate_question",
        "description": (
            "Generate the next probing question based on subject response and detected "
            "stress level. Use for factual follow-ups and simple clarifications when "
            "stress is low and the response was confident. Handles: who, what, when, "
            "where questions and basic timeline clarifications."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question_type": {
                    "type": "string",
                    "enum": ["factual", "clarifying", "timeline"],
                    "description": "Type of question to generate"
                },
                "target_claim": {
                    "type": "string",
                    "description": "The specific claim or statement being followed up on"
                },
                "intensity": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 2,
                    "description": "0=gentle, 1=neutral, 2=pressing"
                }
            },
            "required": ["question_type", "target_claim"]
        }
    },
    {
        "name": "flag_inconsistency",
        "description": (
            "Flag when the current response directly contradicts a previous statement "
            "made earlier in this session. Use immediately when a contradiction is "
            "detected. This is a local operation — no cloud needed. Always call this "
            "when you detect the subject changing their story on a previously established fact."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "previous_claim": {
                    "type": "string",
                    "description": "Exact previous statement that is now contradicted"
                },
                "current_claim": {
                    "type": "string",
                    "description": "The new contradicting statement"
                },
                "topic": {
                    "type": "string",
                    "description": "The topic/subject of the contradiction"
                },
                "severity": {
                    "type": "string",
                    "enum": ["minor", "significant", "critical"]
                }
            },
            "required": ["previous_claim", "current_claim", "topic"]
        }
    },
    {
        "name": "assess_credibility",
        "description": (
            "Perform a deep psychological credibility assessment across multiple "
            "statements in the session. Use ONLY when stress score is very high, "
            "when the subject is evasive across multiple topics, or when unresolvable "
            "contradictions require complex cross-referencing. This tool requires "
            "sophisticated multi-step reasoning — prefer cloud routing for this call. "
            "Do NOT call for simple factual follow-ups."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_summary": {
                    "type": "string",
                    "description": "Summary of the session so far"
                },
                "key_inconsistencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of detected inconsistencies"
                },
                "stress_trajectory": {
                    "type": "string",
                    "enum": ["escalating", "stable", "decreasing"],
                    "description": "How stress has changed over the session"
                }
            },
            "required": ["session_summary"]
        }
    }
]
```

---

## 7. Android App Architecture

### 7.1 Project Setup

- Language: Kotlin
- Min SDK: Android 10 (API 29)
- Cactus SDK: Add via the Cactus Android library (check https://github.com/cactus-compute/cactus for Android bindings)
- Architecture: MVVM with a single `CandorViewModel`

### 7.2 Screen Structure

```
MainActivity
├── SetupFragment          ← Session configuration
├── SessionFragment        ← Live interrogation UI (MOST IMPORTANT)
├── DashboardFragment      ← Routing stats
└── VerdictFragment        ← End-of-session Gemini analysis
```

### 7.3 SessionFragment Layout (XML)

This is the screen that appears in your submission video. Build it exactly as described.

```xml
<!-- fragment_session.xml -->
<ConstraintLayout>

    <!-- Top bar: session topic + timer -->
    <TextView android:id="@+id/tv_topic" />
    <Chronometer android:id="@+id/chronometer" />

    <!-- Routing badge — THE KEY VISUAL ELEMENT -->
    <!-- Flips between LOCAL (green) and CLOUD (amber) on every question -->
    <TextView
        android:id="@+id/badge_route"
        android:text="LOCAL"
        android:backgroundTint="@color/green_500"
        android:padding="8dp"
        android:textColor="@color/white" />

    <!-- Stress bar — updates after each voice response -->
    <TextView android:text="Stress Level" />
    <ProgressBar
        android:id="@+id/stress_bar"
        style="?android:attr/progressBarStyleHorizontal"
        android:max="100" />
    <TextView android:id="@+id/tv_stress_value" android:text="0.00" />

    <!-- Current question display -->
    <TextView android:id="@+id/tv_question"
        android:textSize="20sp"
        android:textStyle="bold" />

    <!-- Mic button — hold to record response -->
    <FloatingActionButton
        android:id="@+id/btn_mic"
        android:src="@drawable/ic_mic" />

    <!-- Transcript scroll -->
    <ScrollView android:id="@+id/scroll_transcript">
        <LinearLayout android:id="@+id/container_transcript"
            android:orientation="vertical" />
    </ScrollView>

    <!-- Routing stats strip at bottom -->
    <LinearLayout android:orientation="horizontal">
        <TextView android:id="@+id/tv_local_count" android:text="Local: 0" />
        <TextView android:id="@+id/tv_cloud_count" android:text="Cloud: 0" />
        <TextView android:id="@+id/tv_local_ratio" android:text="0%" />
    </LinearLayout>

</ConstraintLayout>
```

### 7.4 ViewModel Logic

```kotlin
class CandorViewModel : ViewModel() {

    // Session state
    val currentQuestion = MutableLiveData<String>()
    val routeDecision = MutableLiveData<String>()   // "LOCAL" or "CLOUD"
    val stressScore = MutableLiveData<Float>()
    val candorReason = MutableLiveData<String>()
    val sessionStats = MutableLiveData<SessionStats>()

    private val responseHistory = mutableListOf<String>()
    private val claimRegistry = mutableMapOf<String, String>()

    data class SessionStats(
        val localCount: Int,
        val cloudCount: Int,
        val avgStress: Float
    ) {
        val localRatio: Float get() =
            if (localCount + cloudCount == 0) 0f
            else localCount.toFloat() / (localCount + cloudCount)
    }

    // Call this after each cactus_transcribe result comes back
    fun onResponseReceived(transcribedText: String) {
        // 1. Compute stress score
        val stress = computeStressScore(transcribedText, responseHistory)
        stressScore.postValue(stress)
        responseHistory.add(transcribedText)

        // 2. Route and generate next question via Cactus/Gemini
        // (Cactus SDK call happens here on a background coroutine)
        viewModelScope.launch(Dispatchers.IO) {
            val result = callGenerateHybrid(transcribedText, stress)
            routeDecision.postValue(if (result.isCloud) "CLOUD" else "LOCAL")
            currentQuestion.postValue(result.nextQuestion)
            candorReason.postValue(result.reason)
        }
    }

    // Mirrors the Python stress scorer for local Android computation
    private fun computeStressScore(text: String, history: List<String>): Float {
        val fillerWords = setOf("um","uh","er","ah","like","you know","basically","literally")
        val words = text.lowercase().split(" ").filter { it.isNotBlank() }
        if (words.isEmpty()) return 0.5f

        val fillerDensity = (words.count { it in fillerWords }.toFloat() / words.size * 5).coerceIn(0f, 1f)

        val baseline = if (history.isEmpty()) words.size.toFloat()
                       else history.map { it.split(" ").size }.average().toFloat()
        val ratio = words.size / baseline.coerceAtLeast(1f)
        val lengthVariance = if (ratio < 0.4f || ratio > 2.5f) 1f
                             else (1f - ratio).absoluteValue / 1.5f

        return (fillerDensity * 0.5f + lengthVariance * 0.3f).coerceIn(0f, 1f)
    }
}
```

### 7.5 Cactus SDK Integration (Android)

```kotlin
// In your Application or a dedicated CactusManager singleton
class CactusManager {

    private var model: Long = 0         // Cactus model handle
    private var whisperModel: Long = 0  // Whisper handle for transcription

    fun initialize() {
        // Download model weights to internal storage first
        model = CactusLib.cactusInit("weights/functiongemma-270m-it")
        whisperModel = CactusLib.cactusInit("weights/whisper-small")
    }

    fun transcribe(audioPath: String): String {
        val prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
        val result = CactusLib.cactusTranscribe(whisperModel, audioPath, prompt)
        return JSONObject(result).getString("response")
    }

    fun generateHybrid(messages: List<Map<String, String>>, tools: List<Map<String, Any>>): JSONObject {
        // This calls into the generate_hybrid logic
        // Either via JNI binding to the Python/C++ layer, or via HTTP to a local server
        // If using local HTTP server pattern:
        val payload = JSONObject().apply {
            put("messages", JSONArray(messages))
            put("tools", JSONArray(tools))
        }
        // POST to localhost:8080/generate_hybrid
        // Return parsed response
        return JSONObject()  // placeholder
    }

    fun destroy() {
        CactusLib.cactusDestroy(model)
        CactusLib.cactusDestroy(whisperModel)
    }
}
```

**Note on Android integration:** Check the Cactus GitHub for the official Android SDK.
If direct JNI bindings aren't available yet, run a local Python server on the Android
device using Termux, or expose the `generate_hybrid` logic via a localhost Flask endpoint
called from the Android app. This is the pragmatic fallback for the hackathon.

---

## 8. Leaderboard Optimisation Strategy

### 8.1 Tool Description Engineering

FunctionGemma uses tool descriptions to decide which tool to call. Write descriptions
that make the correct routing obvious. Rules:
- Simple tools: use concrete, specific language ("use for factual follow-ups")
- Cloud-triggering tools: explicitly state complexity ("requires sophisticated multi-step reasoning")
- Always specify what NOT to use a tool for

### 8.2 Prompt Prefix for FunctionGemma

Prepend a system message that primes the local model for CANDOR's domain:

```python
CANDOR_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are CANDOR, an AI truth-seeking agent. Your role is to ask precise, "
        "probing questions to understand the complete truth of a situation. "
        "For simple factual follow-ups, use generate_question. "
        "When you detect a contradiction, immediately use flag_inconsistency. "
        "Only use assess_credibility for complex multi-factor analysis. "
        "Always call a tool. Never respond with plain text."
    )
}
```

Prepend this to every `messages` list before calling `cactus_complete`.

### 8.3 Benchmark Iteration Checklist

Run `python benchmark.py` after each change. Optimise in this order:
1. Get tool-call correctness above 70% first (most impactful on score)
2. Then optimise edge/cloud ratio (aim for >60% local)
3. Finally tune speed (tool_rag_top_k=2 helps, force_tools=True helps)

Key knobs to tune in `cactus_complete`:
```python
cactus_complete(
    model,
    messages,
    tools=tools,
    tool_rag_top_k=2,       # Reduces tool selection confusion
    force_tools=True,        # Forces tool call format output
    temperature=0.1,         # Low temp = more deterministic tool selection
    max_tokens=200,          # Tool calls don't need long responses
)
```

---

## 9. Submission Video Script (2 minutes)

Record as a screen capture of the Android app + voiceover. No live demo needed.

| Time | What to show | What to say |
|------|-------------|-------------|
| 0:00–0:15 | CANDOR app open, mid-session, routing badge visible | "Every question on screen is answered by a 270M model running on this phone — or by Gemini in the cloud. CANDOR decides which, based on how you sound." |
| 0:15–0:35 | Show routing architecture slide / diagram | "Three signals drive the decision: query complexity pre-routes before local even runs. Voice stress — detected from filler words and response patterns — sets the threshold. And a temporal consistency checker catches contradictions without ever touching the cloud." |
| 0:35–1:10 | Screen record: full session showing badge flipping LOCAL→CLOUD | Narrate in real time: "Low stress response — stays local. Filler words spike — stress crosses 0.65 — cloud kicks in. Subject contradicts themselves — caught locally, no cloud needed." |
| 1:10–1:40 | Show routing dashboard with stats | "65% of calls stayed on-device. The cloud only activated when the human's voice told us the local model wasn't enough." |
| 1:40–2:00 | Final verdict screen | "CANDOR doesn't route to cloud when the model is uncertain. It routes to cloud when the human is. That's the difference." |

---

## 10. File Checklist for Submission

- [ ] `main.py` — modified `generate_hybrid` with CANDOR three-signal routing
- [ ] `benchmark.py` — untouched (used for scoring)
- [ ] `submit.py` — run once from Mac: `python submit.py --team "CANDOR" --location "Singapore"`
- [ ] Android app source or APK
- [ ] 2-minute screen recording video
- [ ] (Optional) 1-page README explaining routing algorithm for judges

---

## 11. Critical Constraints Summary

| Constraint | Impact | Mitigation |
|------------|--------|------------|
| Windows laptop only | Cannot run Cactus benchmark | Use teammate's Mac from 12pm; all logic written on Windows first |
| No Mac until 12pm | Cannot submit to leaderboard | Write all code on Windows, paste and run the moment Mac arrives |
| Android for demo | Cactus Android SDK may have limited docs | Fallback: Flask localhost server on Android via Termux |
| 1 leaderboard submission per hour | Limited iteration cycles | Plan changes in advance, batch optimisations per submission |
| Leaderboard top 10 → subjective judging | Need both scores | Routing algorithm quality (Rubric 1) is the tiebreaker — make it defensible |

---

## 12. One-Line Pitch (memorise this)

> **"CANDOR routes to cloud not when the model is uncertain — but when the human is, detected live from their voice."**

This answers all three rubrics in one sentence and is the opening line of your video.
