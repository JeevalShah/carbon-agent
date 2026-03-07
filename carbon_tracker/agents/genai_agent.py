from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class GenAIResponse:
    answer: str
    provider: str  # "openai", "gemini", or "local"
    used_openai: bool = False  # Track if OpenAI was used for UI feedback


def _build_lane_prompt(lane_row: Dict[str, Any]) -> str:
    """
    Prompt template for lane insight generation.
    """
    lane = lane_row.get("lane", "Unknown")
    carbon_intensity = lane_row.get("carbon_intensity", lane_row.get("avg_carbon_intensity", "Unknown"))
    load_factor = lane_row.get("avg_load_factor", lane_row.get("load_factor", "Unknown"))
    distance = lane_row.get("avg_distance_km", lane_row.get("distance_km", "Unknown"))

    return f"""You are a sustainability analytics assistant for road freight logistics.

Lane: {lane}
Carbon intensity (kg CO2 per tonne-km): {carbon_intensity}
Average load factor: {load_factor}
Average distance (km): {distance}

Explain:
1. Why emissions are high (or could be high) on this lane
2. Operational issues that may be driving emissions
3. Sustainability recommendations that are practical for road freight operators

Be specific, action-oriented, and concise. Use bullet points where helpful.
"""


def _fallback_answer(user_query: str, lane_row: Optional[Dict[str, Any]] = None) -> str:
    """
    Deterministic fallback so the prototype still works without an API key.
    """
    parts = []
    if lane_row:
        parts.append(
            f"Lane `{lane_row.get('lane','Unknown')}` looks high-impact because it combines distance, payload (tonne-km), and fuel-related factors."
        )
        parts.append(
            "- **Common drivers**: long average distance, low load factor (empty running), inefficient vehicle choice, congestion/idle time, suboptimal routing."
        )
        parts.append(
            "- **Recommendations**: improve consolidation to raise load factor, reduce deadhead miles, plan routes to avoid peak congestion, maintain tires/engine, upgrade to more efficient vehicles, and set lane-level emission KPIs."
        )
        parts.append(
            f"- **Quick read**: carbon intensity is `{lane_row.get('carbon_intensity', lane_row.get('avg_carbon_intensity','NA'))}` kg CO2/tonne-km."
        )
    else:
        parts.append("I can help explain emissions drivers and recommendations.")
        parts.append(
            "- **Typical causes**: long distance, low load factor, inefficient vehicles, detours, traffic/idle, poor consolidation."
        )
        parts.append(
            "- **Typical fixes**: consolidate loads, optimize routes, reduce empty returns, upgrade vehicles, driver training, preventive maintenance."
        )
    parts.append("")
    parts.append(f"Your question: {user_query}")
    return "\n".join(parts)


def _call_openai_chat(system_prompt: str, user_prompt: str, model: str, api_key: str) -> str:
    """
    Wrapper for OpenAI chat completion. Returns response text or raises on failure.
    """
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )
    return (resp.choices[0].message.content or "").strip()


def _call_gemini_chat(system_prompt: str, user_prompt: str, model: str, api_key: str) -> str:
    """
    Wrapper for Gemini chat completion via google.genai (new package).
    """
    try:
        # Try the new google.genai package first
        import google.genai as genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=system_prompt + "\n\nUser: " + user_prompt,
        )
        return response.text.strip()
    except ImportError:
        pass
    
    # Fallback to deprecated google.generativeai with updated model name
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    # Use the correct model name - try gemini-2.0-flash or gemini-2.0-flash-lite
    model_name = model if model else "gemini-2.0-flash"
    
    m = genai.GenerativeModel(model_name)
    prompt = f"{system_prompt}\n\nUser:\n{user_prompt}"
    resp = m.generate_content(prompt)
    return (resp.text or "").strip()


def _get_api_keys() -> tuple[Optional[str], Optional[str]]:
    """
    Get API keys from environment variables with debugging info.
    Returns (openai_key, gemini_key)
    """
    # Hardcoded API key for development/testing - replace with your key
    GEMINI_API_KEY = "AIzaSyAcir5WkcIXNnpKfW7j4gw82azakNTlXRs"
    
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    
    # Debug: Print which keys are found (without revealing the actual key)
    if openai_key:
        print(f"[DEBUG] OpenAI key found: {openai_key[:8]}...")
    else:
        print("[DEBUG] OpenAI key NOT found in environment")
        
    if gemini_key:
        print(f"[DEBUG] Gemini key found: {gemini_key[:8]}...")
    else:
        print("[DEBUG] Gemini key NOT found in environment")
        
    return openai_key, gemini_key


def generate_lane_insight(
    lane_data: Dict[str, Any] | pd.Series,
    user_query: str = "Provide insights for this lane.",
    model: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> GenAIResponse:
    """
    Generate a natural-language lane explanation using the Gemini API.

    - Works with lane_data as dict or a pandas Series row
    - Falls back to a local, deterministic explanation if Gemini is unavailable
    """
    lane_row = dict(lane_data) if isinstance(lane_data, pd.Series) else dict(lane_data)

    openai_key, gemini_key = _get_api_keys()
    
    # Allow override via api_key parameter (for OpenAI)
    if api_key:
        openai_key = api_key
        
    system_prompt = "You are a helpful sustainability analytics assistant."
    prompt = _build_lane_prompt(lane_row) + f"\nUser question: {user_query}"

    # Prefer Gemini if configured, otherwise fall back to OpenAI, then local.
    if gemini_key:
        try:
            text = _call_gemini_chat(system_prompt, prompt, model=model, api_key=gemini_key)
            if text:
                return GenAIResponse(answer=text, provider="gemini", used_openai=False)
        except Exception as e:
            error_msg = str(e)
            print(f"[DEBUG] Gemini error: {e}")
            # Check for quota error and provide helpful message
            if "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                return GenAIResponse(
                    answer="⚠️ **Google API Quota Exceeded**\n\nYour Google Gemini API key has exceeded its free tier quota. Options:\n1. Wait for quota to reset (usually daily)\n2. Enable billing at https://aistudio.google.com/app/apikey for higher limits\n3. Use a different API key with available quota\n\nThe app will fall back to local responses for now.",
                    provider="gemini",
                    used_openai=False
                )

    if openai_key:
        try:
            text = _call_openai_chat(system_prompt, prompt, model="gpt-4o-mini", api_key=openai_key)
            if text:
                return GenAIResponse(answer=text, provider="openai", used_openai=True)
        except Exception as e:
            print(f"[DEBUG] OpenAI error: {e}")

    text = _fallback_answer(user_query, lane_row=lane_row)
    return GenAIResponse(answer=text + "\n\n⚠️ Note: No API key found. Set GOOGLE_API_KEY or OPENAI_API_KEY environment variable for AI responses.", provider="local", used_openai=False)


def answer_general_query(
    user_query: str,
    lane_df: Optional[pd.DataFrame] = None,
    model: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> GenAIResponse:
    """
    Answer general user queries like:
      - "Which lanes have the highest emissions?"
      - "Why is Mumbai-Pune a high emission lane?"

    If lane_df is provided, the function tries to ground answers in computed analytics.
    """
    lane_row = None
    if lane_df is not None and "lane" in lane_df.columns:
        q = user_query.lower()
        # Simple grounding: if a lane name appears in the query, pick that lane row.
        for lane in lane_df["lane"].astype(str).unique().tolist():
            if lane.lower() in q:
                lane_row = lane_df[lane_df["lane"].astype(str) == lane].iloc[0].to_dict()
                break

        # If user asked for "highest emissions", pick top rows as context.
        if lane_row is None and ("highest" in q or "top" in q) and ("emission" in q or "emissions" in q):
            top = lane_df.sort_values("total_emissions_per_lane", ascending=False).head(5)
            lines = []
            for _, r in top.iterrows(): 
                lines.append(
                    f"- {r['lane']}: {float(r['total_emissions_per_lane']):,.0f} kg CO2, "
                    f"intensity {float(r['carbon_intensity']):.4f} kg/tonne-km"
                )
            return GenAIResponse(
                answer="Top emission lanes (from your data):\n" + "\n".join(lines),
                used_openai=False,
            )

    if lane_row is not None:
        return generate_lane_insight(lane_row, user_query=user_query, model=model, api_key=api_key)

    # If no specific lane was matched, try Gemini first, then OpenAI, then local fallback.
    openai_key, gemini_key = _get_api_keys()
    
    # Allow override via api_key parameter
    if api_key:
        openai_key = api_key
        
    system_prompt = "You are a helpful sustainability analytics assistant for road freight."

    # Prefer Gemini if configured
    if gemini_key:
        try:
            text = _call_gemini_chat(system_prompt, user_query, model=model, api_key=gemini_key)
            if text:
                return GenAIResponse(answer=text, provider="gemini", used_openai=False)
        except Exception as e:
            error_msg = str(e)
            print(f"[DEBUG] Gemini error: {e}")
            # Check for quota error and provide helpful message
            if "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                return GenAIResponse(
                    answer="⚠️ **Google API Quota Exceeded**\n\nYour Google Gemini API key has exceeded its free tier quota. Options:\n1. Wait for quota to reset (usually daily)\n2. Enable billing at https://aistudio.google.com/app/apikey for higher limits\n3. Use a different API key with available quota\n\nThe app will fall back to local responses for now.",
                    provider="gemini",
                    used_openai=False
                )

    if openai_key:
        try:
            text = _call_openai_chat(system_prompt, user_query, model="gpt-4o-mini", api_key=openai_key)
            if text:
                return GenAIResponse(answer=text, provider="openai", used_openai=True)
        except Exception as e:
            print(f"[DEBUG] OpenAI error: {e}")

    return GenAIResponse(
        answer=_fallback_answer(user_query, lane_row=None) + "\n\n⚠️ Note: No API key found. Set GOOGLE_API_KEY or OPENAI_API_KEY environment variable for AI responses.", 
        provider="local", 
        used_openai=False
    )

