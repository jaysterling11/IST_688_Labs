import asyncio, time
import requests
import streamlit as st
from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool, set_default_openai_client

MODEL = "gpt-5-mini"
DEFAULT_LOCATION = "Syracuse, NY"

def get_current_weather(location):
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


@function_tool
def get_weather(location: str = DEFAULT_LOCATION) -> dict:
    """Get the current weather and today's forecast for one location.

    Returns temperature and feels-like temperature (F), a short description
    of conditions, humidity (%), wind speed (mph), UV index, the highest
    chance of rain today (%), and today's high and low (F). If the lookup
    fails, returns a dict with an "error" key instead.

    Args:
        location: A city ("Lima, Peru"), US zip code ("13244"), airport
            code ("SYR"), or landmark. Defaults to Syracuse, NY.
    """

    try:
        return get_current_weather(location)
    except Exception as e:
        return {"error": str(e)}


weather_agent = Agent(
    name="Weather agent",
    instructions=(
        "You are the weather reporter on a 'What to Wear' team. Always call "
        f"get_weather for the location you are given (use '{DEFAULT_LOCATION}' "
        "if none is given). Then summarize the facts in 3 to 5 short lines: "
        "current temperature and feels-like, conditions, today's high and low, "
        "chance of rain, wind, and UV index. Report facts only. Do not give "
        "clothing or activity advice. If the tool returns an error, or the "
        "location is clearly not a real place (for example 'purple banana'), "
        "say that the location could not be found and do not report weather."
    ),
    tools=[get_weather],
    model=MODEL,
)

stylist_agent = Agent(
    name="Stylist agent",
    instructions=(
        "You are a practical stylist. You receive a weather summary and "
        "recommend what to wear today: base layers, outerwear, footwear, and "
        "accessories such as an umbrella, hat, sunglasses, or sunscreen. Give "
        "a short bulleted list and tie each item to a weather fact (for "
        "example 'feels like 38F' or '60% chance of rain'). Use only the "
        "weather facts you were given. Do not suggest activities."
    ),
    model=MODEL,
)

activity_agent = Agent(
    name="Activity agent",
    instructions=(
        "You are a local activity planner. You receive a weather summary and "
        "suggest 3 or 4 outdoor activities that suit today's conditions, each "
        "with a one-line reason. If rain, cold, heat, wind, or high UV makes "
        "being outside unpleasant, also suggest one indoor backup and the "
        "best time of day to go out. Use only the weather facts you were "
        "given. Do not give clothing advice."
    ),
    model=MODEL,
)

manager_agent = Agent(
    name="Manager",
    instructions=(
        "You coordinate a team that answers 'What should I wear and do "
        "today?'. Follow these steps:\n"
        "1. Call weather_report with the user's location (use "
        f"'{DEFAULT_LOCATION}' if none is given).\n"
        "2. If the weather report says the location could not be found, tell "
        "the user, ask for a real city or zip code, and stop. Do not call the "
        "other tools.\n"
        "3. Otherwise, pass the full weather summary to outfit_advice and to "
        "activity_ideas.\n"
        "4. Combine the results into one friendly answer with three short "
        "sections: Weather, What to wear, What to do. Keep the specialists' "
        "advice; do not invent new advice or weather facts of your own."
    ),
    tools=[
        weather_agent.as_tool(
            tool_name="weather_report",
            tool_description=(
                "Looks up today's weather for a location and returns a short "
                "factual summary (no advice). Input: the location to check."
            ),
        ),
        stylist_agent.as_tool(
            tool_name="outfit_advice",
            tool_description=(
                "Recommends what to wear today. Input: the weather summary "
                "from weather_report."
            ),
        ),
        activity_agent.as_tool(
            tool_name="activity_ideas",
            tool_description=(
                "Suggests outdoor activities (and an indoor backup if needed) "
                "for today. Input: the weather summary from weather_report."
            ),
        ),
    ],
    model=MODEL,
)


single_agent = Agent(
    name="Single agent",
    instructions=(
        "You are a helpful assistant that gives clothing and outdoor activity "
        "suggestions based on current weather. Always use the get_weather "
        "tool to check conditions before giving advice. If no location is "
        f"given, use '{DEFAULT_LOCATION}' as the default. If the tool returns "
        "an error, or the location is clearly not a real place, say the "
        "location could not be found and ask for a real city or zip code. "
        "Otherwise answer in three short sections: Weather (the key facts), "
        "What to wear (a bulleted list tied to the weather), and What to do "
        "(3 or 4 outdoor activities, plus an indoor backup if conditions are "
        "poor)."
    ),
    tools=[get_weather],
    model=MODEL,
)

@st.cache_resource
def get_event_loop():
    return asyncio.new_event_loop()


def run(agent, prompt):
    start = time.perf_counter()
    result = get_event_loop().run_until_complete(Runner.run(agent, prompt))
    return result, time.perf_counter() - start

if 'openai_async_client' not in st.session_state:
    try:
        openai_api_key = st.secrets["openai_api_key"]
        st.session_state.openai_async_client = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None
    except Exception:
        st.session_state.openai_async_client = None


if st.session_state.openai_async_client is not None:
    set_default_openai_client(st.session_state.openai_async_client)

st.title("Lab 9: The What to Wear Crew")
st.write(
    "The Lab 5 bot, split into a team: a weather agent, a stylist, and an "
    "activity planner, coordinated by a manager. Switch to the single agent "
    "to compare cost and speed."
)
st.sidebar.header("Settings")
st.sidebar.write(f"Model: {MODEL}")

mode = st.radio(
    "Who should answer?",
    ["Team (manager + 3 specialists)", "Single agent"],
    horizontal=True,
)
location_input = st.text_input("Where are you?", placeholder=f"e.g. {DEFAULT_LOCATION}")

if "runs" not in st.session_state:
    st.session_state.runs = []

if st.button("Get advice"):
    if st.session_state.openai_async_client is None:
        st.error("OpenAI API key was not found. Please add openai_api_key to Streamlit secrets.")
        st.stop()

    location = location_input.strip() or DEFAULT_LOCATION
    agent = manager_agent if mode.startswith("Team") else single_agent

    with st.spinner("The team is working on it (20 to 40 seconds)..."
                    if agent is manager_agent else "Thinking..."):
        try:
            result, seconds = run(agent, f"What should I wear and do today in {location}?")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    with st.chat_message("assistant"):
        st.write(result.final_output)

    usage = result.context_wrapper.usage
    with st.expander("Behind the scenes"):
        st.write(f"Model requests: {usage.requests} · Tokens: {usage.total_tokens:,} · "
                 f"Time: {seconds:.1f}s · Answered by: {result.last_agent.name}")
        for item in result.new_items:
            if item.type == "tool_call_item":
                st.write(f"{item.agent.name} called {item.raw_item.name}: "
                         f"{item.raw_item.arguments}")
            elif item.type == "tool_call_output_item":
                st.write(f"result: {str(item.output)[:300]}")
            elif item.type == "handoff_output_item":
                st.write(f"handoff: {item.source_agent.name} -> {item.target_agent.name}")

    st.session_state.runs.append({
        "Mode": "Team" if agent is manager_agent else "Single agent",
        "Location": location,
        "Model requests": usage.requests,
        "Total tokens": usage.total_tokens,
        "Seconds": round(seconds, 1),
    })

if st.session_state.runs:
    st.subheader("Run history")
    st.dataframe(st.session_state.runs, hide_index=True)

# Part D: 
# a. The manager called weather_report first. After the weather summary came back, it called outfit_advice and activity_ideas, usually in the same turn, so those two ran in parallel. The get_weather call does not appear in the list, because it happens inside the weather agent’s own run rather than the manager’s.
# b. The manager sent the stylist a summary, not the full forecast from the tool. The stylist only receives the “input” text that the manager wrote when it called outfit_advice. The manager model wrote that text by paraphrasing the summary that the weather agent had written from the tool’s JSON.
# c. The other model calls happened inside the specialist agents. Each agent-as-tool runs as its own Runner.run with its own model calls, and the SDK adds those calls to the same usage counter. The weather agent makes two requests, one to call get_weather and one to write the summary, and the stylist and activity agents make one request each. The manager also makes one request per turn: one to call weather_report, one to call the other two tools, and one to write the final answer.