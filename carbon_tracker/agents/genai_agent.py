from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import google.generativeai as genai


@dataclass
class GenAIResponse:
    answer: str
    provider: str


def _build_lane_prompt(lane_row: Dict[str, Any]) -> str:

    lane = lane_row.get("lane", "Unknown")
    carbon_intensity = lane_row.get("carbon_intensity", "Unknown")
    load_factor = lane_row.get("avg_load_factor", "Unknown")
    distance = lane_row.get("avg_distance_km", "Unknown")

    return f"""
You are a sustainability analytics assistant for freight logistics.

Lane: {lane}
Carbon intensity (kg CO2 per tonne-km): {carbon_intensity}
Average load factor: {load_factor}
Average distance (km): {distance}

Explain:
1. Why emissions might be high
2. Operational reasons
3. Sustainability improvements

Use short bullet points.
"""


def _fallback_answer(user_query: str, lane_row: Optional[Dict[str, Any]] = None) -> str:

    if lane_row:

        return f"""
Lane **{lane_row.get('lane','Unknown')}** may have higher emissions because:

• Long travel distance  
• Low vehicle load factor  
• Diesel truck usage  
• Route inefficiencies  

Recommended improvements:

• Increase load consolidation  
• Reduce empty return trips  
• Improve route planning  
• Use fuel-efficient vehicles
"""

    return """
Freight emissions are usually driven by:

• Long distance transport  
• Low truck utilization  
• Inefficient routing  
• Older diesel fleets  

Improvement strategies:

• Load consolidation
• Route optimization
• Reduce empty miles
• Fleet upgrades
"""

def _call_gemini(prompt: str, api_key: str) -> str:

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-1.5-flash")

    response = model.generate_content(prompt)

    return response.text.strip()

def _get_gemini_key():
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def generate_lane_insight(
    lane_data: Dict[str, Any] | pd.Series,
    user_query: str = "Explain this lane."
) -> GenAIResponse:

    lane_row = dict(lane_data) if isinstance(lane_data, pd.Series) else lane_data

    gemini_key = _get_gemini_key()

    prompt = _build_lane_prompt(lane_row) + f"\nUser question: {user_query}"

    if gemini_key:
        try:

            text = _call_gemini(prompt, gemini_key)

            return GenAIResponse(
                answer=text,
                provider="gemini"
            )

        except Exception:
            pass

    return GenAIResponse(
        answer=_fallback_answer(user_query, lane_row),
        provider="local"
    )

def answer_general_query(
    user_query: str,
    lane_df: Optional[pd.DataFrame] = None
) -> GenAIResponse:

    lane_row = None

    if lane_df is not None and "lane" in lane_df.columns:

        q = user_query.lower()

        for lane in lane_df["lane"].astype(str).unique():

            if lane.lower() in q:

                lane_row = lane_df[lane_df["lane"] == lane].iloc[0].to_dict()

                break

    if lane_row:

        return generate_lane_insight(lane_row, user_query)

    gemini_key = _get_gemini_key()

    if gemini_key:
        try:

            text = _call_gemini(user_query, gemini_key)

            return GenAIResponse(
                answer=text,
                provider="gemini"
            )

        except Exception:
            pass

    return GenAIResponse(
        answer=_fallback_answer(user_query),
        provider="local"
    )
