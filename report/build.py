import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/aradavtyan/nn-dynamics-agentic-ai-hw2"


def build():
    fonts = Path("/usr/share/fonts/truetype/dejavu")
    pdfmetrics.registerFont(TTFont("DejaVu", str(fonts / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuBold", str(fonts / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVuBold", italic="DejaVu", boldItalic="DejaVuBold")
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "DejaVu"
        style.leading = 13
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].spaceAfter = 8
    styles["Heading1"].leading = 23
    story = []

    def p(text):
        story.append(Paragraph(text, styles["BodyText"]))

    def heading(text):
        story.append(Paragraph(text, styles["Heading1"]))
        story.append(Spacer(1, 12))

    def table(rows, widths):
        cells = [[Paragraph(escape(str(cell)), styles["BodyText"]) for cell in row] for row in rows]
        item = Table(cells, colWidths=widths, repeatRows=1)
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#a7b4c1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(item)
        story.append(Spacer(1, 12))

    sessions = []
    for name in ("recording_03", "recording_02"):
        folder = ROOT / "results/live" / name
        record = json.loads((folder / "run_record.json").read_text())
        evidence = json.loads((folder / "demo_evidence.json").read_text())
        if evidence["completion_status"] != "completed":
            raise ValueError(f"{name} is not a completed run")
        experiments = [e for e in record["experiments"] if e["status"] == "completed"]
        sessions.append((name, folder, record, evidence, experiments))

    def code_link(label, path):
        url = f"{REPOSITORY}/blob/main/{path}"
        p(f'<b>{label}:</b> <link href="{url}" color="blue">{escape(url)}</link>')

    heading("Homework 2")
    p("PhotonicsAI — Agentic Systems with Hardware")
    p("<b>Student: Ara Davtyan</b><br/>Prepared: 13 October 2026")
    p("This report describes the human-in-the-loop system and two completed live focusing sessions. The language model chose the experiment settings through OpenRouter. All light measurements came from the simulated bench.")
    p(f'<b>Repository:</b> <link href="{REPOSITORY}" color="blue">{REPOSITORY}</link>')
    p("<b>Report status:</b> Task 2 has measured live results and camera images. Task 1 demonstration runs have not been performed yet. The lecture comparison and the screen-recording attachment still need to be completed before submission.")
    heading("Task 1. Human-in-the-loop")
    p("The base system has four roles: orchestrator, scientist, general agent and executor. They share one dialogue. The orchestrator reads it and makes a model call to choose the next destination, question and reason. The graph allows at most eight orchestrator decisions. The base source is kept in task6/agentic_system.py.")
    p("A fifth destination, human, asks for clarification or approval. The graph is compiled with an InMemorySaver checkpointer. LangGraph interrupt() pauses it on a thread ID. Command(resume=reply) continues the same thread. The reply enters the shared dialogue with role human and is available to the following turn. This checkpointer keeps state while the process stays open; it is not persistent across restarts.")
    p("For an unclear request, the question asks which interpretation to follow. Before running code, the executor prepares the action, and approval names that exact action and code hash. A refusal adds a message that the action was rejected and sends control back to the orchestrator without running the tool. Final measurement reporting also needs approval. The implementation refuses to overwrite existing result folders.")
    p("<b>Run (a): not yet performed.</b> The request is ‘Report the RMSE from the Silverbox work.’ My Homework 1 reference has training RMSE 0.002332454 V and test RMSE 0.008616101 V, so the answer depends on the intended split. The required evidence is the actual question, my typed reply and the next dialogue message. These have not been collected yet.")
    p("<b>Run (b): not yet performed.</b> The hard-decision demo asks to run a Silverbox computation. The required evidence is the first approval question and refusal, the message proving that the refused action did not run, then the second question and acceptance. No example dialogue is presented as a real run.")
    code_link("Task 1 source", "hw2/task1_hitl.py")
    story.append(PageBreak())
    heading("Task 2. Method and autonomous run")
    p("The simulated path is laser → binary-amplitude DMD → scattering medium → camera. The DMD changes amplitude only. The 16×16 DMD provides 256 input pixels; the camera has 32×32 output pixels. One fixed complex Gaussian transmission matrix T describes the scattering medium (seed 7). The recorded ideal intensity is I = |T Ein|².")
    p("Noise is added to every shot: Poisson shot noise with 1000 photons per intensity unit and Gaussian read noise with standard deviation 0.01. Negative noisy intensities are clipped to zero. A repeated shot of the same pattern uses a fresh noise sample.")
    p("The search starts with a random binary pattern. A grouping factor g makes each macropixel g×g DMD pixels, giving (16/g)² controls. After each candidate transmission, the tool measures the target and four nearby background strips. It flips the chosen number of macropixels and retains candidates with higher measured enhancement. Only camera measurements guide the search; the optimizer does not inspect T.")
    p("Let F be mean intensity in the target. For each background strip, calculate its mean, then average the four strip means to obtain B. Enhancement is η = F/B, with a small denominator floor for numerical stability. Independent shots validate both the starting and retained final patterns. The report uses the mean and sample standard deviation of these validation enhancements, rather than the highest noisy value seen during search.")
    p("<b>User request:</b> ‘Focus the beam through the simulated scattering medium.’ No pixel size, region size or iteration count was entered during either run. The orchestrator chose roles through model calls, the scientist suggested experiments, and the executor chose the tool settings. Only the executor called FocusingSession.call. The graph kept the returned measurements and selected the completed experiment with the largest validated final enhancement in each session.")
    p("Both sessions used the model route openrouter/auto. The saved logs identify that route, but do not identify each underlying provider model; no specific provider model is claimed. Each session was limited to 3 completed experiments, 4000 camera acquisitions, 8 orchestrator decisions and 30 model calls.")
    code_link("Task 2 agent", "hw2/task2_agent.py")
    code_link("Focusing tool", "hw2/focusing_tool.py")
    story.append(PageBreak())
    heading("Task 2. Experiment settings and results")
    p("The main session is recording_03. The earlier completed session recording_02 is also included. All six experiments used medium seed 7 and pattern seed 7, and a background gap of 2 camera pixels. Coordinates below are zero-based. A target entry gives (row, column), then height×width.")
    for name, folder, record, evidence, experiments in sessions:
        p(f"<b>{name}</b>: {len(experiments)} experiments, {record['num_orchestrator_decisions']} orchestrator decisions, {record['total_camera_acquisitions']} camera acquisitions, {evidence['runtime_s']:.2f} seconds. Status: completed.")
        rows = [["Call", "Macro / controls", "Target", "BG width", "Steps / flips", "Avg / n", "Cap / shots"]]
        for i, e in enumerate(experiments, 1):
            a, result = e["arguments"], e["summary"]
            g = a["grouping_factor"]
            rows.append([i, f"{g}×{g} / {(16//g)**2}", f"({a['focus_row']},{a['focus_col']}) {a['focus_height']}×{a['focus_width']}", a["background_width"], f"{a['iterations']} / {a['flips_per_proposal']}", f"{a['averaging']} / {a['validation_count']}", f"{a['max_acquisitions']} / {result['camera_acquisitions']}"])
        table(rows, [30, 80, 88, 48, 65, 55, 100])
        rows = [["Call", "Initial η", "Final η ± SD", "Contrast gain", "Focus gain", "BG suppression"]]
        for i, e in enumerate(experiments, 1):
            result = e["summary"]
            rows.append([i, f"{result['validated_initial_enhancement_mean']:.4f}", f"{result['validated_final_enhancement_mean']:.4f} ± {result['validated_final_enhancement_std']:.4f}", f"{result['enhancement_gain']:.3f}×", f"{result['focus_intensity_gain']:.3f}×", f"{result['background_suppression_factor']:.3f}×"])
        table(rows, [30, 65, 125, 80, 75, 91])
    p("Avg is the number of shots averaged per measurement. n is the number of independent validation measurements for each starting/final pattern. Cap is the maximum allowed camera acquisitions per experiment; shots is the actual count. SD describes variation across validation measurements, not a confidence interval. Contrast gain = final η / initial η. Focus gain = final F / initial F. Background suppression = initial B / final B.")
    for name, folder, record, evidence, experiments in sessions:
        story.append(PageBreak())
        heading(f"Task 2. Camera images — {name}")
        p("Left: starting pattern. Right: retained final pattern. Each image averages the independent validation measurements. The outlines mark the target and nearby background regions. Each before/after pair uses a shared colour scale; scales may differ between experiments.")
        for i, e in enumerate(experiments, 1):
            result = e["summary"]
            p(f"<b>Call {i}</b> — enhancement {result['validated_initial_enhancement_mean']:.4f} → {result['validated_final_enhancement_mean']:.4f}")
            images = []
            for filename in ("initial_camera.png", "final_camera.png"):
                path = folder / "artifacts" / e["run_name"] / filename
                item = Image(str(path))
                item.drawHeight = 175 * item.imageHeight / item.imageWidth
                item.drawWidth = 175
                images.append(item)
            story.append(Table([images], colWidths=[233, 233]))
            story.append(Spacer(1, 6))
    story.append(PageBreak())
    heading("Task 2. Findings and lecture comparison")
    main = sessions[0]
    best = max(main[4], key=lambda e: e["summary"]["validated_final_enhancement_mean"])
    result = best["summary"]
    p(f"The main session selected call {main[4].index(best)+1}. Its validated final enhancement was {result['validated_final_enhancement_mean']:.4f} ± {result['validated_final_enhancement_std']:.4f}. Local contrast improved by {result['enhancement_gain']:.3f} times, while target intensity increased by {result['focus_intensity_gain']:.3f} times. Background suppression contributed a factor of {result['background_suppression_factor']:.3f}. About {100*result['share_of_log_gain_from_focus_brightening']:.1f}% of the log contrast gain came from target brightening and {100*result['share_of_log_gain_from_background_suppression']:.1f}% from background suppression. The large enhancement therefore mainly reflects a darker nearby background.")
    p("In recording_02, call 1 improved contrast but reduced target intensity (focus gain 0.853). This shows why enhancement and absolute target brightness must be reported separately. The later calls brightened their targets. More controls gave better contrast in these trials, but iteration counts, averaging and sometimes region sizes also changed. These experiments do not isolate the effect of macropixel size or establish an optimal setting.")
    p("The images show local target-to-background improvement. The target is a region rather than one camera pixel, and no whole-frame brightest-pixel test was performed. All experiments used one fixed random medium. Choosing the largest validation enhancement among several experiments may favour a lucky validation set; a fresh held-out measurement would give a stronger check.")
    p("<b>Connection to the lecture method.</b> The simulation follows the laser–DMD–scattering-medium–camera path and the amplitude-pattern search described in the assignment. Random patterns produce speckle, and changing groups of DMD pixels improves contrast in the selected region. This is a comparison of the method, not a measured comparison with the lecture’s focusing result.")
    p("<b>Lecture-result comparison still needed.</b> I do not have the lecture image or its enhancement value, so I cannot state whether this result is brighter, sharper or higher in enhancement. A direct comparison needs the lecture’s target size, background definition and result. A random Gaussian medium and simulated camera noise also differ from a physical bench.")
    p("<b>Saved evidence.</b> results/live/recording_03/ contains dialogue.json, experiments.csv, run_record.json, demo_evidence.json, final_report.txt and the images under artifacts/. The earlier session is saved in results/live/recording_02/. The screen recording should show a model decision, the executor’s tool call and a completed experiment. It must be supplied in the repository or attached to the same email as this PDF; the video file has not been verified for this report.")
    p("<b>Remaining submission evidence:</b> the two actual Task 1 dialogues, the lecture-result comparison, the screen-recording attachment and confirmation that the linked code has been pushed. The Task 2 values in this report are from completed live sessions, not the offline demonstration.")
    output = ROOT / "report/homework2.pdf"
    SimpleDocTemplate(str(output), title="Homework 2", author="Ara Davtyan", leftMargin=48, rightMargin=48, topMargin=44, bottomMargin=44).build(story)
    print(output)


if __name__ == "__main__":
    build()
