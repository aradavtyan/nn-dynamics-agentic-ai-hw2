# Homework 2 — Agents and simulated focusing

Ara Davtyan · [Repository](https://github.com/aradavtyan/nn-dynamics-agentic-ai-hw2)

This project has two tasks. In Task 1, an agent asks a person before making an
unclear or important decision. In Task 2, an agent uses a simulated camera and
DMD to make a bright spot. The written report is in `report/homework2.pdf`.
Homework 1 is used as a read-only reference; its files are unchanged.
This is the planned Homework 2 repository; these changes have not been pushed.

## Files

- `hw2/task1_hitl.py`: the graph, questions, approval and resume.
- `hw2/demo_task1.py`: the unclear Silverbox request.
- `hw2/demo_task1_hard.py`: refuse a coding action, then approve it.
- `hw2/simulated_bench.py`: the DMD, scattering medium and noisy camera.
- `hw2/focusing_optimizer.py`: change pixels and keep better patterns.
- `hw2/focusing_tool.py`: run an experiment and save its images and measurements.
- `hw2/task2_agent.py`: the four roles and the focusing tool.
- `hw2/demo_task2_live.py`: run with a real language model.
- `tests/`: offline checks for the graph, approval and simulator.
- `results/offline_demo/`: measured simulator results with scripted model replies.
- `results/live/recording_02/` and `recording_03/`: completed live agent sessions,
  with experiment tables, dialogue and camera images used in the report.
- `report/`: the report source and local PDF.
- `task6/agentic_system.py`: the shared four-role base used by both tasks.
- `results/silverbox_reference.json`: your selected Homework 1 model's training
  and test errors, used only as context for the clarification example.

## Setup

Use Python 3.11 or newer. Run these commands from `homework_2`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Try the simulator

```bash
python -m hw2.task2_agent --offline-demo
```

This needs no API key. The model replies are scripted. The camera images and
measurements come from the simulator. This checks that the parts work together;
it is not the real agent run required for submission.

The saved example has two experiments. Their final enhancements are 13.77 and
70.78. Enhancement is mean brightness in the target divided by mean brightness
in nearby background regions. It can improve because the target gets brighter,
the background gets darker, or both.

## Task 6 — Shared agent source

The Homework 1 four-role agent source is kept in `task6/agentic_system.py`.
Homework 2 imports it for both tasks. To run the base agent on its own:

```bash
python -m task6.agentic_system --data ../homework_1/data/SNLS80mV.mat
```

This needs `OPENAI_API_KEY`. The default request asks about the Silverbox RMSE;
pass `--task path/to/request.txt` to use a different request.
The old run logs were removed during cleanup and have not been restored.

## Task 1 — Ask a person

Set `OPENAI_API_KEY` in your shell, then run:

```bash
python -m hw2.demo_task1
python -m hw2.demo_task1_hard --data ../homework_1/data/SNLS80mV.mat
```

The data path uses the local Homework 1 file without changing it. For a separate
checkout, pass your own `SNLS80mV.mat` path.

The orchestrator chooses the next role using the model. A human question pauses
the LangGraph thread. The reply resumes that same thread and enters the shared
dialogue. A refused coding action does not run. Approval applies to the exact
prepared code. The graph also asks before reporting a final measurement.

## Task 2 — Focus the beam

```bash
python -m hw2.demo_task2_live --max-api-calls 30
```

The user gives the goal. The agent chooses the pixel size, target and background
regions, iteration count and other settings. Only the executor calls the tool.
The bench computes `abs(T @ Ein) ** 2` and adds shot noise and camera read noise.
The search starts from a random binary pattern, flips some pixels and keeps
changes that improve enhancement. Separate camera shots check the final result.

Each run saves its dialogue, experiment settings, table and camera images in a
new folder under `runs/`. These scratch runs are ignored by Git. Copy the
chosen live evidence into `results/live/` before submitting. Record the terminal
while the agent decides, calls the tool and runs an experiment; keep that video
in `results/live/` or attach it to the submission email.

## Written submission

```bash
python report/build.py
```

The report uses the title, student name and clean layout of Homework 1.
The report includes both completed live Task 2 sessions and their camera images;
recording_03 is the main session. To finish the assignment, add the real Task 1
questions, replies and next messages, and the recording.
The report compares the best simulated enhancement of 9.93 with the lecture’s
approximate experimental best of 33, noting the limits of this comparison.
Send one PDF as requested in the assignment. The PDF is built locally and is
ignored by Git, like Homework 1's submission PDF.

## OpenRouter

Set `OPENROUTER_API_KEY` to use OpenRouter instead of OpenAI. If both keys are
set, OpenRouter is used. Enter the key privately before starting the recording:

```bash
read -s -p "OpenRouter API key: " OPENROUTER_API_KEY
export OPENROUTER_API_KEY
echo
clear
python -m hw2.demo_task2_live --model openrouter/auto --workdir results/live/recording_02 --max-api-calls 30
```

Use a new output folder for each run. `openrouter/auto` chooses a model through
OpenRouter. You can instead pass a specific provider/model ID from your account.
OpenRouter usage has its own credit balance. No key is saved in the project.
The same key and `--model` option work with the two Task 1 demos.
