52 Card Pickup — Multi‑Agent Simulation

This repository contains a simple but complete demonstration of a
multi‑agent system built with the LangGraph
 library. The project
implements a deterministic 52 Card Pickup game as a stateful
graph with four cooperating agents. It is meant to serve as a
“hello world” for multi‑agent orchestration: the focus is on clearly
illustrating roles, interactions, shared state and scaling rather than
LLM integration or flashy visuals.

Agents and responsibilities

The application uses four logical agents, each represented by a node in
a LangGraph state graph:

Scatter Agent – creates a standard deck of 52 cards and assigns
each card a random (x, y) position within a 10×10 grid. Every
card is marked as unpicked. This agent runs once at the start of
the simulation.

Timer Agent (start) – records a high‑precision timestamp
immediately before the pickup phase begins. Timing uses
time.perf_counter() for accuracy.

Pickup Agent(s) – one or more agents that pick up cards.

All agents start at (0, 0) and repeatedly select the nearest
unpicked card within their assigned region of the grid. After
picking up a card they move to that card’s position. Their
progress is tracked by marking cards as picked_up=True and
recording the picked_up_by identifier.

The number of pickup agents is configurable. With 1 agent the
entire deck is collected by a single worker. With 2 agents the
grid is split into left and right halves (based on whether
x < 5), and with 4 agents the grid is split into four quadrants
(based on x and y being less than or greater than 5). Each
agent operates only on cards in its region.

For more than one agent the heavy work of nearest‑neighbour search
runs concurrently in separate processes, using
concurrent.futures.ProcessPoolExecutor to sidestep Python’s
global interpreter lock. Within each region, greedy selection is
used to minimise travel distance.

Timer Agent (stop) – records a timestamp once all pickup
agents have finished.

Verifier Agent – checks that exactly 52 unique cards exist,
ensures every card is marked as picked up, and that no duplicates
exist. It writes a PASS/FAIL message into the state.

Although LangGraph supports conditional edges and return types to
create loops, this example keeps looping
logic inside the pickup node itself for clarity. The graph only
specifies the order of high‑level stages: scatter → start timer →
pickup → stop timer → verify. Node functions accept and return the
shared state, and edges determine
sequencing.

Running the simulation
Installation

Ensure Python 3.11 or later is installed.

Install dependencies into a virtual environment:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Usage

To run the simulation and observe scaling behaviour across different
numbers of pickup agents, execute the script directly:

python card_pickup.py

The script will perform ten trials each with 1, 2 and 4 pickup agents.
Randomness is seeded for reproducibility (seed = 42 for the first run
and incremented thereafter). After all runs complete, a summary table
is printed showing the average, best and worst pickup times along with
whether the verifier passed in each configuration. An example output
might look like this:

| Agents | Avg Time (s) | Best (s) | Worst (s) | Verifier |
|--------|--------|--------|--------|--------|
| 1 | 0.0123 | 0.0111 | 0.0135 | 10/10 ✓ |
| 2 | 0.0087 | 0.0079 | 0.0094 | 10/10 ✓ |
| 4 | 0.0061 | 0.0055 | 0.0067 | 10/10 ✓ |

Smaller average times when using more agents demonstrate how
parallelism can speed up independent sub‑tasks. The verifier column
indicates how many runs passed the final checks (a check mark denotes
all runs passed).

How it works

Internally the card_pickup.py script defines a state schema using
TypedDict and registers Python functions as nodes in a
langgraph.StateGraph. A brief overview of the workflow:

Scatter – builds a fresh deck of 52 cards, each with random
coordinates in [0, 10). All cards are flagged as unpicked.

Timer start – writes the current value of time.perf_counter()
into the state.

Pickup – determines how many pickup agents to run (from
state['pickup_agents']), partitions the deck into regions and
launches one worker per region. Each worker repeatedly finds the
nearest unpicked card, marks its index and continues until all
regional cards are collected. Once all workers return, cards in
the shared state are updated accordingly.

Timer stop – writes the end time into the state.

Verify – validates that there are 52 unique suit–rank pairs
and that every card is marked as picked up. A PASS/FAIL message is
stored in state['result'].

The graph is compiled once per run and invoked with an initial state.
LangGraph handles passing the state between nodes and ensures the
workflow executes in the prescribed order.

Files

card_pickup.py – entry point containing the agents, state definition
and scaling experiment. Running this script will perform the
experiments and print a summary table.

requirements.txt – pinned dependency list. Only
langgraph
 is required.

License

This project is provided for educational purposes. Feel free to use
it as a starting point for your own explorations into LangGraph and
multi‑agent workflows.
