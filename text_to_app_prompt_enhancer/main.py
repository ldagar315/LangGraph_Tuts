import mesop as me
import mesop.labs as mel
import backend

@me.stateclass
class AppState:
    user_idea: str = ""
    problem_to_solve: str = ""
    technicals_output: str = ""
    mvp_features_output: str = ""
    ui_output: str = ""
    ux_output: str = ""
    is_loading: bool = False
    error_message: str = ""

def on_idea_input(e: me.InputEvent):
    state = me.state(AppState)
    state.user_idea = e.value

def on_problem_input(e: me.InputEvent):
    state = me.state(AppState)
    state.problem_to_solve = e.value

def on_generate_click(e: me.ClickEvent):
    state = me.state(AppState)
    state.is_loading = True
    state.error_message = "" # Clear previous errors
    # Clear previous outputs
    state.technicals_output = ""
    state.mvp_features_output = ""
    state.ui_output = ""
    state.ux_output = ""
    yield # Update UI to show loading state

    try:
        print("Invoking LangGraph...")
        # Prepare the initial state for the graph
        initial_state: state = {
            "user_input": state.user_idea,
            "problem_to_solve": state.problem_to_solve,
            "technicals": "",
            "mvp_features": "",
            "ui_design": "",
            "ux_design": "",
            "final_draft": ""
        }
        # Run the graph
        # Use .invoke for synchronous execution suitable for a single request/response
        final_state = backend.final_graph(initial_state)
        print("LangGraph invocation complete.")
        print("Final State:", final_state)

        # Update Mesop state with results from the final graph state
        state.technicals_output = final_state.get("technicals", "No technicals generated.")
        state.mvp_features_output = final_state.get("mvp_features", "No MVP features generated.")
        state.ui_output = final_state.get("ui_design", "No UI design generated.")
        state.ux_output = final_state.get("ux_design", "No UX design generated.")

    except Exception as err:
        print(f"An error occurred during graph execution: {err}")
        state.error_message = f"An error occurred: {err}"
    finally:
        # Always ensure loading is set to false
        state.is_loading = False
        yield # Update UI to show results/errors and reset button

@me.page(
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=["https://google.github.io"]
    ),
    path="/",
    title="MVP Idea Generator"
)
def page():
    state = me.state(AppState)

    with me.box(style=me.Style(display="flex", flex_direction="column", gap="15px", padding=me.Padding.all(20))):
        me.text("MVP Idea Generator", type="headline-4")
        me.text("Enter your app idea and the core problem it solves to generate MVP details.")

        # Input Section
        with me.box(style=me.Style(display="flex", flex_direction="column", gap="10px")):
            me.textarea(
                label="Your App Idea",
                value=state.user_idea,
                on_input=on_idea_input,
                style=me.Style(width="100%"),
                disabled=state.is_loading
            )
            me.textarea(
                label="Problem It Solves (Optional)",
                value=state.problem_to_solve,
                on_input=on_problem_input,
                style=me.Style(width="100%"),
                disabled=state.is_loading
            )

        # Button and Loading Indicator
        with me.box(style=me.Style(display="flex", align_items="center", gap="10px")):
            me.button(
                "Generate MVP Plan",
                on_click=on_generate_click,
                type="raised",
                color="primary",
                disabled=state.is_loading or not state.user_idea # Disable if loading or no idea input
            )
            if state.is_loading:
                me.progress_spinner(diameter=30) # Show spinner when loading

        # Display potential errors
        if state.error_message:
             me.text(state.error_message, style=me.Style(color="red"))

        me.divider()

        # Output Section
        me.text("Generated MVP Details", type="headline-5")
        with me.box(style=me.Style(display="grid", grid_template_columns="1fr 1fr", gap="20px", margin=me.Margin(top=15))):
             # Box 1: MVP Features
             with me.box(style=me.Style(border=me.Border.all(me.BorderSide(style="solid", width=1, color="#ccc")), padding=me.Padding.all(15), border_radius=8)):
                  me.text("MVP Features", type="subtitle-1", style=me.Style(margin=me.Margin(bottom=10)))
                  me.markdown(state.mvp_features_output if state.mvp_features_output else "_Click 'Generate' to see MVP features_") # Use markdown for lists

             # Box 2: Technicals
             with me.box(style=me.Style(border=me.Border.all(me.BorderSide(style="solid", width=1, color="#ccc")), padding=me.Padding.all(15), border_radius=8)):
                  me.text("Technicals (Tech Stack)", type="subtitle-1", style=me.Style(margin=me.Margin(bottom=10)))
                  me.markdown(state.technicals_output if state.technicals_output else "_Click 'Generate' to see the tech stack_") # Use markdown for lists

             # Box 3: UX Design
             with me.box(style=me.Style(border=me.Border.all(me.BorderSide(style="solid", width=1, color="#ccc")), padding=me.Padding.all(15), border_radius=8)):
                  me.text("UX Design (User Flow)", type="subtitle-1", style=me.Style(margin=me.Margin(bottom=10)))
                  me.markdown(state.ux_output if state.ux_output else "_Click 'Generate' to see the UX design_") # Use markdown for lists

             # Box 4: UI Design
             with me.box(style=me.Style(border=me.Border.all(me.BorderSide(style="solid", width=1, color="#ccc")), padding=me.Padding.all(15), border_radius=8)):
                  me.text("UI Design (Layout & Elements)", type="subtitle-1", style=me.Style(margin=me.Margin(bottom=10)))
                  me.markdown(state.ui_output if state.ui_output else "_Click 'Generate' to see the UI design_") # Use markdown for lists
