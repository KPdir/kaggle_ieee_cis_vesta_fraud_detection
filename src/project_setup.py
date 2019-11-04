import os

# hide warnings
import warnings
warnings.filterwarnings('ignore')

# Data directory paths
raw_data_dir = os.path.relpath("../data/raw/")
interim_data_dir = os.path.relpath("../data/interim/")
processed_data_dir = os.path.relpath("../data/processed/")
submissions_dir = os.path.relpath("../data/submissions/")

# Models directory paths
models_dir = os.path.relpath("../models/")

# Toggle cell visibility
from IPython.display import HTML
import random

def hide_toggle(for_next=False):
    """
    Toggle code visibility in Jupyter. 
    
    Source:
    https://stackoverflow.com/questions/31517194/how-to-hide-one-specific-cell-input-or-output-in-ipython-notebook
    """
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Show/Hide Code'  # Text shown on toggle link
    target_cell = this_cell  # Target cell to control with toggle
    js_hide_current = ''  # Bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += ''
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1,2**64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current, 
        toggle_text=toggle_text
    )

    return HTML(html)