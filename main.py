import streamlit as st
import hashlib

from lib.graph import create_participant_graphs
from lib.file_tools import create_participants, load_participants

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
if "master_archive" not in st.session_state:
    st.session_state.master_archive = {}  # Stores { p_name: p_obj }
if "participants" not in st.session_state:
    st.session_state.participants = []  # List of p_name strings
if "participant_objs" not in st.session_state:
    st.session_state.participant_objs = {}  # {p_name: p_obj}
if "loaded_file_ids" not in st.session_state:
    st.session_state.loaded_file_ids = set()  # Set of uploaded file IDs already processed
if "pause" not in st.session_state:
    st.session_state.pause = False
if "graphs" not in st.session_state:
    st.session_state.graphs = {}  # {p_name: [graph_dict, ...]}
if "selected_graph" not in st.session_state:
    st.session_state.selected_graph = None


# Callback function: Runs when files change, before the UI draws
def handle_file_upload():
    # Look at the widget state directly using a key
    uploaded_files = st.session_state.csv_uploader

    if uploaded_files:
        # Check if any file in the uploader is actually new
        has_new_files = False
        for file in uploaded_files:
            file_bytes = file.getvalue()
            file_id = hashlib.md5(file_bytes).hexdigest()
            if file_id not in st.session_state.loaded_file_ids:
                has_new_files = True
                break

        # If there are new files to process, lock the UI right now
        if has_new_files:
            st.session_state.pause = True

st.title("CSV Data Analysis Interface")

left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    st.subheader("File Upload & Participants")

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload CSV File(s)",
        type=["csv"],
        accept_multiple_files=True,
        key="csv_uploader",
        on_change=handle_file_upload,
    )

    # IMMEDIATE & ONE-TIME LOADING PARTICIPANT
    if uploaded_files and st.session_state.pause:
        # Optional visual indicator that background loading is happening
        with st.spinner("Importing new participants..."):
            new_files = []
            for file in uploaded_files:
                file_bytes = file.getvalue()
                file_id = hashlib.md5(file_bytes).hexdigest()

                if file_id not in st.session_state.loaded_file_ids:
                    new_files.append(file)
                    # Add to the set immediately so it's tracked right away
                    st.session_state.loaded_file_ids.add(file_id)

            # 2. Bulk process the objects only if there are new files
            if new_files:
                p_objs = create_participants(new_files)
                for p_obj in p_objs:
                    p_name = p_obj.get_name()

                    # Save to BOTH active state and the master memory archive
                    st.session_state.participant_objs[p_name] = p_obj
                    st.session_state.master_archive[p_name] = p_obj

                    if p_name not in st.session_state.participants:
                        st.session_state.participants.insert(0, p_name)

        # Ingestion finished: Lift the pause lock immediately so the button can react
        st.session_state.pause = False
        st.rerun()

    # Process Button (Generates Graphs for loaded participants)
    is_empty = len(st.session_state.participant_objs) == 0
    button_disabled = st.session_state.pause or is_empty

    if st.button("Process CSV Files", type="primary", disabled=button_disabled):
        with st.status("Processing data...", expanded=False) as status:
            load_participants(st.session_state.participant_objs.values())
            st.session_state.graphs = create_participant_graphs(
                st.session_state.participant_objs.values()
            )
            status.update(label="Complete!", state="complete")
        st.success("Graphs processed successfully!")

    # Handle Active Participant
    def handle_participant_removal():
        # Read the current selection from the widget's key
        current_active = st.session_state.active_participants_multiselect

        # Find who was removed
        removed_items = set(st.session_state.participants) - set(current_active)

        if removed_items:
            for p_name in removed_items:
                # 1. Safely clear visualizations
                if st.session_state.graphs:
                    st.session_state.graphs.pop(p_name, None)

                if (
                        st.session_state.selected_graph
                        and p_name in st.session_state.selected_graph.get("id", "")
                ):
                    st.session_state.selected_graph = None

                # 2. Hard deletion option: If you remove from participant_objs,
                # you MUST clear the file id tracking so it can be re-uploaded.
                st.session_state.participant_objs.pop(p_name, None)

                # NOTE: If your load_participants_id function exposes the file_id or file_name
                # on the p_obj, clear it from loaded_file_ids here:
                # f_id = st.session_state.participant_objs[p_name].get_file_id()
                # st.session_state.loaded_file_ids.discard(f_id)

            # Update the master tracking list
            st.session_state.participants = current_active


    # 3. Active Participant UI
    st.subheader("Active Participants")

    # The options are EVERYTHING we have ever loaded successfully
    all_loaded_names = list(st.session_state.master_archive.keys())

    if all_loaded_names:
        active_participants = st.multiselect(
            "Add or remove participants from the active session:",
            options=all_loaded_names,
            default=st.session_state.participants  # Defaults to what is currently active
        )

        # Sync states instantly if the user changes selections
        if set(active_participants) != set(st.session_state.participants):
            # 1. Find who was removed to clean up graphs
            removed_items = set(st.session_state.participants) - set(active_participants)
            for p_name in removed_items:
                st.session_state.graphs.pop(p_name, None)
                if st.session_state.selected_graph and p_name in st.session_state.selected_graph.get("id", ""):
                    st.session_state.selected_graph = None

            # 2. Rebuild active objects directly from memory archive (O(1) speed - instant!)
            st.session_state.participant_objs = {
                name: st.session_state.master_archive[name] for name in active_participants
            }
            st.session_state.participants = active_participants
            st.rerun()

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