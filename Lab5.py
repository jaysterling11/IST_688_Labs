import requests
import json
import streamlit as st
from openai import OpenAI

DEFAULT_LOCATION = "Syracuse, NY"
MODEL = "gpt-4o-mini"

def get_current_weather(location):
    url = f'https://wttr.in/{location}?format=j1'
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise Exception(f'wttr.in error: status {response.status_code}')
    try:
        data = response.json()
    except ValueError:
        # unknown locations come back as plain text, not JSON
        raise Exception(f'Could not find a location named {location}')

    current = data['current_condition'][0]
    today = data['weather'][0]
    chance_of_rain = max(int(h.get("chanceofrain", 0)) for h in today["hourly"])

    return {
        "location": location,
        "temperature_F": float(current["temp_F"]),
        "feels_like_F": float(current["FeelsLikeF"]),
        "description": current["weatherDesc"][0]["value"],
        "humidity_pct": int(current["humidity"]),
        "wind_speed_mph": float(current["windspeedMiles"]),
        "uv_index": int(current.get("uvIndex", 0)),
        "chance_of_rain_pct": chance_of_rain,
        "today_max_F": float(today["maxtempF"]),
        "today_min_F": float(today["mintempF"]),
    }

weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": (
            "Get the current weather and today's forecast summary for a "
            "given location, to be used for clothing and activity advice."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "The city, zip code, airport code, or landmark to "
                        f"get weather for. Default to '{DEFAULT_LOCATION}' "
                        "if the user did not specify a location."
                    ),
                }
            },
            "required": ["location"],
        },
    },
}
 
 
def call_weather_tool(tool_call) -> str:
    args = json.loads(tool_call.function.arguments)
    location = args.get("location")
    try:
        weather_data = get_current_weather(location)
    except Exception as e:
        weather_data = {"error": str(e)}
    return json.dumps(weather_data)
 
 
def get_outfit_advice(client: OpenAI, location_input: str) -> str:
    user_prompt = (
        f"What should I wear today and what outdoor activities would be "
        f"appropriate, for the location: {location_input}?"
        if location_input.strip()
        else (
            "What should I wear today and what outdoor activities would be "
            "appropriate? I didn't specify a location, so use the default."
        )
    )
 
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that gives clothing and outdoor "
                "activity suggestions based on current weather. Always use "
                "the get_current_weather tool to check conditions before "
                f"giving advice. If no location is given, use '{DEFAULT_LOCATION}' "
                "as the default."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
 
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=[weather_tool],
        tool_choice="auto",
    )
    response_message = response.choices[0].message
    messages.append(response_message.to_dict())
 
    tool_calls = response_message.tool_calls
    if not tool_calls:
        return response_message.content
 
    for tool_call in tool_calls:
        if tool_call.function.name == "get_current_weather":
            result = call_weather_tool(tool_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": result,
                }
            )
 
    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return final_response.choices[0].message.content

if 'openai_client' not in st.session_state:
    try:
        openai_api_key = st.secrets["openai_api_key"]
        st.session_state.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
    except Exception:
        st.session_state.openai_client = None

st.title("👕 What to Wear Bot")
 
st.write(
    "Enter a city, zip code, airport code, or landmark below and this bot "
    "will check today's weather (via wttr.in) and suggest what to wear and "
    "which outdoor activities make sense. Under the hood, the weather lookup "
    "is provided to the LLM as a tool call — the model decides when it needs "
    "the weather and requests it, rather than us fetching it ahead of time."
)
 
st.sidebar.header("Chatbot Settings")
st.sidebar.write(f"Model: {MODEL}")
 
location_input = st.text_input(
    "Where are you?", placeholder=f"e.g. {DEFAULT_LOCATION}"
)
 
if st.button("Get suggestions"):
    if st.session_state.openai_client is None:
        st.error("OpenAI API key was not found. Please add openai_api_key to Streamlit secrets.")
        st.stop()
 
    with st.spinner("Checking the weather and thinking..."):
        try:
            advice = get_outfit_advice(st.session_state.openai_client, location_input)
            st.markdown(advice)
        except Exception as e:
            st.error(f"Something went wrong: {e}")