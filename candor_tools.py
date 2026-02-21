"""
CANDOR-specific tools for truth-seeking and adaptive questioning.
Use these tools for the voice demo, not for benchmark.py compatibility.
"""

CANDOR_TOOLS = [
    {
        "name": "generate_question",
        "description": (
            "Generate the next probing question based on subject response and detected "
            "stress level. Use for factual follow-ups and simple clarifications when "
            "stress is low and response was confident. Handles: who, what, when, "
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
            "Flag when current response directly contradicts a previous statement "
            "made earlier in this session. Use immediately when contradiction detected. "
            "This is a local operation - no cloud needed. Always call when subject "
            "changes their story on a previously established fact."
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
            "Perform deep psychological credibility assessment across multiple "
            "statements in the session. Use ONLY when stress score is very high, "
            "when subject is evasive across multiple topics, or when unresolvable "
            "contradictions require complex cross-referencing. This tool requires "
            "sophisticated multi-step reasoning - prefer cloud routing for this call. "
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
