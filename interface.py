import streamlit as st
from lib.graph import create_participant_graphs

# Attempt importing custom module (with fallback for testing)
try:
    from lib.file_tools import load_participant
except ImportError:
    class DummyParticipant:
        def __init__(self, file):
            self.file = file
            self.name = file.name.replace(".csv", "").strip()

        def get_name(self):
            return self.name


    def load_participant(file):
        return DummyParticipant(file)

# Download Feature
import io
import datetime as dt
def get_figure_bytes(fig, dpi=300) -> bytes:
    """Converts a matplotlib figure into PNG bytes for downloading."""
    buf = io.BytesIO()
    # dpi=300 ensures publication-quality resolution
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    return buf.getvalue()

# Page Setup
st.set_page_config(page_title="Research Graph Dashboard", layout="wide")

# Session State Initialization
if "participants" not in st.session_state:
    st.session_state.participants = []  # List of p_name strings
if "participant_objs" not in st.session_state:
    st.session_state.participant_objs = {}  # {p_name: p_obj}
if "loaded_file_ids" not in st.session_state:
    st.session_state.loaded_file_ids = set()  # Set of uploaded file IDs already processed
if "graphs" not in st.session_state:
    st.session_state.graphs = {}  # {p_name: [graph_dict, ...]}
if "selected_graph" not in st.session_state:
    st.session_state.selected_graph = None

st.title("CSV Data Analysis Interface")

left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    st.subheader("File Upload & Participants")

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload CSV File(s)",
        type=["csv"],
        accept_multiple_files=True
    )

    # IMMEDIATE & ONE-TIME LOADING
    if uploaded_files:
        for file in uploaded_files:
            # Use file.file_id (or unique identifier fallback) to check if already loaded
            file_id = getattr(file, "file_id", f"{file.name}")

            if file_id not in st.session_state.loaded_file_ids:
                # Runs ONLY ONCE per uploaded file
                p_obj = load_participant(file)
                p_name = p_obj.get_name()

                # Re-position if already in list (move to front)
                if p_name in st.session_state.participants:
                    st.session_state.participants.remove(p_name)

                st.session_state.participants.insert(0, p_name)
                st.session_state.participant_objs[p_name] = p_obj

                # Track file ID so load_participant won't run again on reruns
                st.session_state.loaded_file_ids.add(file_id)


    # Process Button (Generates Graphs for loaded participants)
    if st.button("Process CSV Files", type="primary"):
        if st.session_state.participant_objs:
            st.session_state.graphs = create_participant_graphs(st.session_state.participant_objs.values())
            st.success("Graphs processed successfully!")
            st.rerun()
        else:
            st.warning("Please upload at least one CSV file first.")

    st.divider()

    # Active Participant Management
    st.subheader("Active Participants")
    if st.session_state.participants:
        active_participants = st.multiselect(
            "Remove participants by clearing them from this list:",
            options=st.session_state.participants,
            default=st.session_state.participants
        )

        # Sync changes if user manually removes someone from the multiselect
        if set(active_participants) != set(st.session_state.participants):
            removed_items = set(st.session_state.participants) - set(active_participants)
            for p_name in removed_items:
                st.session_state.graphs.pop(p_name, None)
                st.session_state.participant_objs.pop(p_name, None)

                if (
                        st.session_state.selected_graph
                        and p_name in st.session_state.selected_graph["id"]
                ):
                    st.session_state.selected_graph = None

            st.session_state.participants = [
                p for p in st.session_state.participants if p in active_participants
            ]
            st.rerun()
    else:
        st.info("Upload CSV files above to view active participants.")

with right_col:
    st.subheader("Previews")

    with st.container(height=480):
        if st.session_state.graphs:
            mini_cols = st.columns(2)
            idx = 0

            for graph_data in st.session_state.graphs["Overall"]:
                with mini_cols[idx % 2]:
                    st.caption(f"**{graph_data['title']}**")
                    st.pyplot(graph_data["fig"], use_container_width=True)

                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("Expand", key=f"btn_exp_{graph_data['id']}"):
                            st.session_state.selected_graph = graph_data
                    with btn_col2:
                        st.download_button(
                            label="Save",
                            data=get_figure_bytes(graph_data["fig"]),
                            file_name=f"{graph_data['id']}_{dt.date.today()}.png",
                            mime="image/png",
                            key=f"btn_dl_{graph_data['id']}"
                        )
                idx += 1

            for p_name in st.session_state.participants:
                if p_name in st.session_state.graphs:
                    for graph_data in st.session_state.graphs[p_name]:
                        with mini_cols[idx % 2]:
                            st.caption(f"**{graph_data['title']}**")
                            st.pyplot(graph_data["fig"], use_container_width=True)

                            if st.button("Expand", key=f"btn_{graph_data['id']}"):
                                st.session_state.selected_graph = graph_data
                        idx += 1
        else:
            st.caption("Minimized graphs will display here once you click 'Process CSV Files'.")

# Bottom Layout: Large Graph View
st.divider()
st.subheader("Graph View")

if st.session_state.selected_graph:
    selected = st.session_state.selected_graph

    col_title, col_close = st.columns([4, 1])
    with col_title:
        st.write(f"### Detailed View: {selected['title']}")
    with col_close:
        if st.button("Close Large View"):
            st.session_state.selected_graph = None
            st.rerun()

    st.pyplot(selected["fig"], use_container_width=True)
else:
    st.info("Click the **'Expand'** button under any mini-graph on the right to project it here.")