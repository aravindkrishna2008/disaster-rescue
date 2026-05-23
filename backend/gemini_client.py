"""
Track 2 owns this file. This stub lets Track 3 run independently.
Replace the body of get_gemini_target() with the real Gemini call.
"""
import os

SURVIVORS = {
    "child": (3.2, -1.5),
    "adult": (-2.1, 4.3),
}


def get_gemini_target(command: str, survivors: dict) -> dict:
    """
    Returns {"target_id": "child"|"adult", "confidence": float, "reason": str}.
    Stub returns "child" always — Track 2 replaces with real Gemini call.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return {"target_id": "child", "confidence": 0.5, "reason": "stub (no GEMINI_API_KEY)"}

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "response_mime_type": "application/json",
            },
        )

        survivor_desc = "\n".join(
            f"- {sid}: position {pos}" for sid, pos in survivors.items()
        )

        prompt = f"""You are controlling a disaster rescue robot.
Survivors in the environment:
{survivor_desc}

Natural language command: "{command}"

Respond with JSON choosing which survivor to rescue next.
Schema: {{"target_id": "child" | "adult", "confidence": 0.0-1.0, "reason": "<one sentence>"}}"""

        response = model.generate_content(prompt)
        import json
        data = json.loads(response.text)
        if data.get("target_id") not in survivors:
            raise ValueError(f"Unknown target_id: {data.get('target_id')}")
        return data

    except Exception as e:
        # Fallback — never crash the demo
        print(f"[gemini_client] fallback triggered: {e}")
        return {"target_id": "child", "confidence": 0.0, "reason": f"fallback: {e}"}
