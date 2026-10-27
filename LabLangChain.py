import os
import requests
import streamlit as st
from langchain.agents import create_agent

DEFAULT_LOCATION = "Syracuse, NY"
MODEL = "openai:gpt-4o-mini"

def get_weather_data(location: str = "Syracuse, NY") -> dict:
    """Get the current weather and today's forecast for one location, to use
    for clothing and outdoor-activity advice.

    Accepts a city name ("Syracuse, NY", "Lima, Peru"), a US zip code
    ("13244"), a 3-letter airport code ("SYR"), or a landmark. For a landmark
    or any place that is not a city, put a ~ in front of it ("~Eiffel Tower").
    Look up one location per call; if the user names several places, call
    this tool once for each. If the user gives no location, use the default
    "Syracuse, NY".

    Returns a dict with: location, temperature_F, feels_like_F, description,
    humidity_pct, wind_speed_mph, uv_index, chance_of_rain_pct, today_max_F,
    today_min_F. If the location cannot be found, returns a dict with an
    "error" key instead.
    """
    try:
        url = f'https://wttr.in/{location}?format=j1'
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            raise Exception(f'wttr.in error: status {response.status_code}')
        try:
            data = response.json()
        except ValueError:
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
    except Exception as e:
        return {"location": location, "error": str(e)}


SYSTEM_PROMPT = (
    "You are a helpful assistant that gives clothing and outdoor "
    "activity suggestions based on current weather. Always use "
    "the get_weather_data tool to check conditions before "
    f"giving advice. If no location is given, use '{DEFAULT_LOCATION}' "
    "as the default."
)


st.title("Lab 6: What to Wear Bot (LangChain)")

st.write(
    "The same What to Wear bot as Lab 5, rebuilt with LangChain. Enter a city, "
    "zip code, airport code, or landmark and the agent will check today's "
    "weather (via wttr.in) and suggest what to wear and which outdoor "
    "activities make sense. LangChain builds the tool schema, runs the tool, "
    "and keeps calling the model until it has an answer."
)

st.sidebar.header("Chatbot Settings")
st.sidebar.write(f"Model: {MODEL}")


try:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
except Exception:
    st.error("OpenAI API key was not found. Please add openai_api_key to Streamlit secrets.")
    st.stop()

agent = create_agent(
    model=MODEL,                
    tools=[get_weather_data],     
    system_prompt=SYSTEM_PROMPT,  
)

location = st.text_input("Where are you?", placeholder=f"e.g. {DEFAULT_LOCATION}")

if st.button("Get advice"):
    with st.spinner("Checking the weather and thinking"):
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": f"What should I wear in {location}?"}]}
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    with st.chat_message("assistant"):
        st.write(result["messages"][-1].content)

    with st.expander("Behind the scenes"):
        for message in result["messages"]:
            st.write(f"**{message.type}**")
            st.write(message.content)
            if message.type == "ai" and message.tool_calls:
                st.write(message.tool_calls)



# PART D ANSWERS (run with "Syracuse, NY")
#
# a. How many messages are there, and what is each one?
#    4 messages:
#      1. human - my question: "What should I wear in Syracuse, NY?"
#      2. ai    - the model asking for the weather tool (content is empty)
#      3. tool  - the result of get_weather_data: the weather dict as JSON text
#      4. ai    - the final clothing/activity advice shown in the chat bubble
#    The system prompt is NOT in the list. create_agent sends it to the model
#    on every call, but it is not stored in result["messages"].
#
# b. Which message is the model asking for the weather tool?
#    Message 2, which is the first "ai" message. Its content is empty because the
#    request is not text: it is in message.tool_calls, e.g.
#      [{"name": "get_weather_data", "args": {"location": "Syracuse, NY"},
#        "id": "call_...", "type": "tool_call"}]
#    The "id" is what links it to the tool message that answers it.
#
# c. In Lab 5, which lines of my code created each of those messages?
#      1. human -> lines 74-82 build the user prompt and line 95 puts it in
#                  the messages list ({"role": "user", ...}). Lines 85-94 also
#                  added the system message by hand.
#      2. ai (tool request) -> lines 98-103 (first chat.completions.create with
#                  tools=[weather_tool]); lines 104-105 append it to messages.
#      3. tool  -> lines 111-121: the loop over tool_calls, call_weather_tool()
#                  (lines 63-70: json.loads the arguments, run the function,
#                  json.dumps the result) and the append of the
#                  {"role": "tool", "tool_call_id": ...} message.
#      4. ai (final answer) -> lines 123-127 (second chat.completions.create).
#                  Lab 5 never appended this one to messages; it just
#                  returned the content.