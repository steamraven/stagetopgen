"""
MIT License

Copyright (c) 2023 Matthew Hawn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


# pyright: reportUnusedFunction=false


import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, SupportsInt, cast

import pandas
import traittypes  # type: ignore Needed to ensure shinylive includes this dep
from htmltools import css
from pandas import NA, DataFrame, isna, notna  # type: ignore  pandas stub is incomplete
from pandas._libs.missing import NAType
from py2vega.functions.string import pad
from shinywidgets import output_widget, reactive_read, register_widget, render_widget

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

# Reactivity requires copying dataframes instead of in-place updates
pandas.options.mode.copy_on_write = True


##########################################################
# tunables

default_datafile = Path(__file__).parent / "stagetop.csv"

name_columns = ["Type", "Name", "Style", "Version"]  # For creating the part name

# sorting main dataframe
sort_columns = [
    "has_qty",
    "Type",
    "Name",
    "Style",
    "PrintTime",
    "Filament",
]
sort_order = [True if c != "has_qty" else False for c in sort_columns]

QtyCallable = Callable[[int, int, str, str, str], int]

# Quantity Calculation Formulas
qty_calculations: dict[str, QtyCallable] = {
    "Frame": lambda w, h, rail, playtile, leg: w * h,  # Simple
    "FrameLock": lambda w, h, rail, playtile, leg: (
        ((w - 1) * h)
        + (w * (h - 1))  # one to connect each frame
        + (
            (2 * w + 2 * h) if rail != "Lite" else 0
        )  # for non-lite rail, one for each frame edge
    ),
    "TileLock": lambda w, h, rail, playtile, leg: 4 * w * h,  # 4 per faame
    "4": lambda w, h, rail, playtile, leg: 4,  # 4 corners to a rectangle
    "Tile": lambda w, h, rail, playtile, leg: (
        (w - 1) * (h - 1)  # Full tiles overlay internal frame joins
    ),
    "TileHalf": lambda w, h, rail, playtile, leg: (
        2 * (w - 2) + 2 * (h - 2)  # Half tiles fill in edges - corners
    ),
    "Rail": lambda w, h, rail, playtile, leg: (
        2 * (w - 1) + 2 * (h - 1)  # edges - corners
    ),
    "LiteRailLock": lambda w, h, rail, playtile, leg: (
        4  # one lock / corner rail
        + 2 * (2 * (w - 1) + 2 * (h - 1))  # two per reg rail
        + (
            4 * (w - 1) * (h - 1) if leg == "Lite - Short" else 0
        )  # four for each internal frame join if using locks as legs
    ),
}

# Style selection controls


@dataclass
class StyleDependency:
    condition_type: str
    condition_style: str
    dependency_type: str
    dependency_style: str


style_dependencies: list[StyleDependency] = [
    StyleDependency("Leg", "Lite - Short", "Rail", "Lite")
]


@dataclass
class StyleInput:
    part_type: str
    select_name: str
    label: str


style_inputs = [
    StyleInput("Rail", "select_rail_style", "Rail Style:"),
    StyleInput("Playtile", "select_playtile_style", "Playtile Style:"),
    StyleInput("Leg", "select_leg_style", "Leg Style:"),
]


########################################################################
# utility functions

# Print Time in CSV and displayed as HH:MM
# Stored in dataframe as just total minutes
# Vega is a subset of python used as expression language by DataGrid


def to_minutes(text: str | None) -> int | None:
    if text is None or not text:
        return None
    split = text.split(":")
    if len(split) != 2:
        return None
    return int(split[0]) * 60 + int(split[1])


def format_minutes(min: float) -> str:
    if isna(min):
        return ""
    min = int(min)
    return str(min // 60) + ":" + str(min % 60).rjust(2, "0")


# must return truthy value
def format_minutes_vega(cell: Any):
    if cell.value:
        return (
            str(int(cell.value / 60)) + ":" + pad(str(cell.value % 60), 2, "0", "left")
        )
    else:
        return "-"


def to_int_nan(value: str | None) -> int | NAType:
    if value is None or value == "":
        return NA
    else:
        return int(value)


# Display "-" instead of NaN for blank numbers"""
# Must return truthy value
def format_empty_nan_vega(cell: Any):
    return cell.value if cell.value else "-"


# Must pass datagrid module as it can only be imported within the server function
def dg_renderers(dg: types.ModuleType) -> dict[str, Any]:
    "Datagrid renderers for time and integers"
    time_renderer = dg.TextRenderer(
        text_value=dg.Expr(
            format_minutes_vega,
        ),
        missing="",
    )
    nan_renderer = dg.TextRenderer(text_value=dg.Expr(format_empty_nan_vega))
    return {
        "PrintTime": time_renderer,
        "ExtPrinTime": time_renderer,
        "Infill": nan_renderer,
        "Filament": nan_renderer,
        "ExtFilament": nan_renderer,
    }


def set_required(tag: ui.Tag) -> ui.Tag:
    "Set the required attr on input tags"
    assert tag.name == "div" and len(tag.children) == 2
    for child_tag in tag.children:
        if isinstance(child_tag, ui.Tag) and child_tag.name == "input":
            child_tag.attrs["required"] = True
    return tag


def check_style_dependencies(input: Inputs) -> str | None:
    "Check style dependencies using table.  Return None for no error, or error string"
    table = {s.part_type: getattr(input, s.select_name)() for s in style_inputs}
    for dep in style_dependencies:
        if table[dep.condition_type] == dep.condition_style:
            if table[dep.dependency_type] != dep.dependency_style:
                return f'{dep.condition_type} style "{dep.condition_style}" requires {dep.dependency_type} style "{dep.dependency_style}"'
    return None


def instr_row(instr: ui.TagChild, *args: ui.TagChild):
    "UI Helper to renter an instruction row"
    return ui.layout_sidebar(
        ui.panel_sidebar(instr, width=3, style=css(height="100%")),
        ui.panel_main(*args, width=9),
    )


def column_auto(*args: ui.TagChild):
    return ui.div(*args, class_="col-auto")


#############################################################################################################
# UI/Frontend


app_ui = ui.page_fluid(
    ui.div(
        ui.h2("Table Creator for the StageTop Modular, 3D Printed Table"),
        ui.a(
            "Kickstarter page",
            href="https://www.kickstarter.com/projects/gutshotgames/stagetop-the-3d-printed-gaming-table",
        ),
        ui.p("Copyright 2023 Matthew Hawn"),
        ui.p(
            "This project is independant and not affiliated with StageTop, GutShotGames, or MyMiniFactory "
        ),
        class_="text-center",
    ),
    instr_row(
        "Step 1: Select datafile or use default",
        ui.input_file("file_base_datafile", "Data File:", placeholder="Select File"),
    ),
    ui.hr(),
    instr_row(
        "Step 2: Generate a basic table",
        ui.row(
            set_required(ui.input_numeric("numeric_width", "Width:", 3)),
            set_required(ui.input_numeric("numeric_height", "Height:", 4)),
        ),
        ui.row(*(ui.input_select(s.select_name, s.label, []) for s in style_inputs)),
        ui.row(ui.output_text("text_style_error")),
        ui.row(ui.input_action_button("button_generate_table", "Generate Table")),
    ),
    ui.hr(),
    instr_row(
        ui.div(
            ui.p("Step 3: Review components"),
            ui.p("You can sort and filter on any column"),
            ui.p("To reset all values, regenerate a table above"),
        ),
        ui.row(
            ui.input_switch("switch_show_qty", "Only Show Qty > 0"),
            ui.input_action_button(
                "button_clear_time_filament",
                "Clear ALL time and filament",
                width="300px",
            ),
        ),
        ui.row(output_widget("widget_grid")),
    ),
    ui.hr(),
    instr_row(
        ui.div(
            "Step 4: Update components",
            ui.HTML("&nbsp;"),
            ui.input_action_link("link_comp_more", "More Info"),
        ),
        ui.row(
            ui.div(
                ui.row("Part Name"),
                ui.row(ui.output_text("text_selection_name")),
                class_="col",
            ),
            column_auto(
                ui.row("Qty"),
                ui.row(
                    set_required(
                        ui.input_numeric(
                            "numeric_selection_qty", None, value=0, width="100px"
                        )
                    )
                ),
            ),
            column_auto(
                ui.row("Print Time"),
                ui.row(ui.input_text("numeric_selection_time", None, width="100px")),
            ),
            column_auto(
                ui.row("Filament (g)"),
                ui.row(
                    ui.input_numeric(
                        "numeric_selection_filament", None, value=0, width="100px"
                    ),
                ),
            ),
            column_auto(
                ui.row(ui.br()),
                ui.row(
                    ui.input_action_button(
                        "button_selection_update", "Update", width="90px"
                    )
                ),
            ),
        ),
    ),
    ui.hr(),
    instr_row(
        "Step 5: Review totals",
        ui.row(
            column_auto(
                "Totals:",
            ),
            column_auto(
                ui.output_text("text_totals_error", inline=True),
            ),
        ),
        ui.row(
            output_widget("widget_totals"),
        ),
    ),
    ui.hr(),
    instr_row(
        "Step 6: Save your data",
        ui.row(ui.download_button("button_download_data", "Download data")),
    ),
    title="Table Creator for StageTop",
)

##################################################################################################
# Backend


def server(input: Inputs, output: Outputs, session: Session):
    # Reactive values
    base_df: reactive.Value[Any | None] = reactive.Value(None)
    qty_df: reactive.Value[Any | None] = reactive.Value(None)

    # Hack to handle reactive updates to selection by server side code
    selections_value: reactive.Value[list[dict[str, int]] | None] = reactive.Value(None)

    # Datagrid
    import ipydatagrid as dg  # must import within server funciont (Session context)

    datagrid = dg.DataGrid(DataFrame())
    datagrid.renderers = dg_renderers(dg)
    datagrid.selection_mode = "row"
    datagrid.layout.height = "300px"
    register_widget("widget_grid", datagrid)  # Link to output_widget

    # Hack to handle selection saving when updating grid
    prev_selections: dict[str, int] | None = None

    @reactive.Calc
    def get_selection() -> tuple[None, None] | tuple[int, int]:
        """
        Get the current single row selection as row, key or None, None

        Reactive Inputs: selections_value
        Other Inputs: datagrid
        """
        selections = selections_value()
        if (
            selections
            and len(selections) == 1
            and selections[0]["r1"] == selections[0]["r2"]
        ):
            row = selections[0]["r1"]
            if len(datagrid._data["data"]):
                visible: DataFrame = datagrid.get_visible_data()
                if row < len(visible.index):
                    key = int(cast(SupportsInt, visible.index[row]))
                    return (row, key)
        return None, None

    ### Step 1: Either select data file or use default

    @reactive.Effect
    def parse_datafile():
        """
        Hande datafile selection.  Called on start

        Reactiive Inputs: file_base_datafile
        Reactive Outputs:  base_df, qty_df, selections_value
        Other Inputs: style_inputs
        Side-effects: style select controls
        """
        if input.file_base_datafile() is not None:
            data_file = input.file_base_datafile()[0]["datapath"]
        else:
            data_file = default_datafile

        df: DataFrame = pandas.read_csv(  # type: ignore pandas stub is incomplete
            data_file,
            dtype={
                "Infill": "Int8",
                "Filament": "Int32",
            },
            converters={
                "PrintTime": to_minutes,
            },
            na_filter=False,
        )

        # Ensure a Qty column
        if "Qty" not in df.columns:
            df.insert(0, "Qty", 0)  # type: ignore  pandas stub is incomplete

        # Update Style selection controls based on data and style_inputs

        def style_choices(part_type: str):
            """Get unique styles for part types that are marked Default"""
            all_choices = [
                set(c.split(","))
                for c in cast(
                    Iterator[str],
                    df.loc[(df["Type"] == part_type) & (df["Default"] != ""), "Style"],
                )
            ]
            choices = set[str].union(*all_choices)
            return sorted(list(choices))

        for s in style_inputs:
            ui.update_select(
                s.select_name,
                choices=style_choices(s.part_type),
                selected=["Standard", "1x1"],
            )

        base_df.set(df)
        qty_df.set(df)

        # clear datagrid selections and trigger
        datagrid.clear_selection()
        selections_value.set(None)

    ### Step 2 - Generate a table

    @output
    @render.text
    def text_style_error():
        """
        Render any errors based on style dependencies

        Reactive Inputs: style select controls
        Other Inputs: style_inputs, style_dependencies
        """
        return check_style_dependencies(input)

    @reactive.Effect
    @reactive.event(input.button_generate_table)
    def generate_table():
        """
        Generate a table based on default qtys for the selected styles

        Reactive Inputs: button_generate_table
        Reactive Outputs: qty_df, selections_value
        Other Inputs: style select controls, numeric_width, numeric_height, qty_calculations, style_inputs, style_dependencies
        """
        df = base_df()
        if df is None or check_style_dependencies(input):
            return

        def filter(row: "pandas.Series[Any]"):
            style = set(cast(str, row["Style"].split(",")))
            return cast(str, row["Default"]) != "" and (
                any(
                    cast(str, row["Type"]) == s.part_type
                    and cast(str, getattr(input, s.select_name)()) in style
                    for s in style_inputs
                )
                or (cast(str, row["Type"]) == "Core")
            )

        args = (
            input.numeric_width(),
            input.numeric_height(),
            *(getattr(input, s.select_name)() for s in style_inputs),
        )

        def calc_qty(row: "pandas.Series[Any]"):
            if not filter(row):
                return 0
            return qty_calculations[cast(str, row["Default"])](*args)

        df = df.assign(Qty=df.apply(calc_qty, axis=1))
        if isna(df["Qty"]).any():  # type: ignore  pandas stub is incomplete
            print("Error in calculation")
            qty_df.set(base_df())
            return
        qty_df.set(df)

        # clear datagrid selections and trigger
        datagrid.clear_selection()
        selections_value.set(None)

    ######## Step 3 - Update Qty

    @reactive.Effect
    def update_datagrid():
        """
        Reactive Input: qty_df, switch_show_qty
        Reactive Ouptut: datagrid.data
        Side-Effects: prev_selections
        """
        df = qty_df()

        if df is None:
            return None

        # Add some useful columns like has_qty and sort
        df = df.assign(
            ExtPrinTime=df["PrintTime"] * df["Qty"],
            ExtFilament=df["Filament"] * df["Qty"],
            has_qty=df["Qty"] > 0,
        ).sort_values(sort_columns, axis=0, ascending=sort_order)

        # Filter to only used rows
        if input.switch_show_qty():
            df = df.loc[df["has_qty"], :]

        # Save selections to reset later
        with reactive.isolate():
            nonlocal prev_selections

            prev_selections = datagrid.selections[0] if datagrid.selections else None

        # update datagrid. Clears selections
        datagrid.data = df

        # if prev_selections:
        #    datagrid.select(prev_selections[0], prev_selections[1], clear_mode="all")

    @reactive.Effect
    def update_selections():
        """
        Handle changes in selection

        Reactive Inputs: datagrid.selections
        Reactive Outputs: selections_value
        Other Inputs: prev_selections
        Side-Effects: datagrid.select
        """
        nonlocal prev_selections
        selections = reactive_read(datagrid, "selections")

        # Hack to resend selections on datagrid data change
        if not selections and prev_selections is not None:
            tmp = prev_selections
            prev_selections = None
            datagrid.select(tmp["r1"], tmp["c1"], clear_mode="all")
            selections_value.set([tmp])
        else:
            selections_value.set(selections)

    @output
    @render.text
    def text_selection_name():
        """
        Generate a unique name
        Reactive Inputs: selections_value, qty_df
        Other Inputs: name_columns
        """
        _, key = get_selection()
        df = qty_df()
        if df is None or key is None:
            return None

        data = df.loc[key, name_columns]
        return "/".join(str(c).strip() for c in data if c)

    @reactive.Effect
    def update_selection_qty():
        """
        Reactive Inputs: selections_value, qty_df
        Side-Effects: numeric_selection_qty
        """
        _, key = get_selection()
        df = qty_df()
        if df is None:
            return

        value = int(df.at[key, "Qty"]) if key is not None else ""
        # update_numeric can take a "" to clear the field. None does not work
        ui.update_numeric("numeric_selection_qty", value=value)  # type: ignore

    @reactive.Effect
    def update_selection_time():
        """
        Reactive Inputs: selections_value, qty_df
        Side-Effects: numeric_selection_time
        """
        _, key = get_selection()
        df = qty_df()
        if df is None:
            return

        value = format_minutes(df.at[key, "PrintTime"]) if key is not None else ""
        ui.update_text("numeric_selection_time", value=value)

    @reactive.Effect
    def update_selection_filament():
        """
        Reactive Inputs: selections_value, qty_df
        Side-Effects: numeric_selection_filament
        """
        _, key = get_selection()
        df = qty_df()
        if df is None:
            return
        value = df.at[key, "Filament"] if key is not None else None
        value = int(value) if value is not None and notna(value) else ""
        # update_numeric can take a "" to clear the field. None does not work
        ui.update_numeric("numeric_selection_filament", value=value)  # type: ignore

    @reactive.Effect
    @reactive.event(input.button_selection_update)
    def button_selection_update():
        """
        Update values in dataframe/datagrid when update button pressed

        Reactive Inputs: button_selection_update
        """
        df = qty_df()
        row, key = get_selection()

        if df is None or row is None:
            return

        if input.numeric_selection_qty() is None:
            return

        df = df.copy()
        df.at[key, "Qty"] = int(input.numeric_selection_qty())
        df.at[key, "PrintTime"] = to_minutes(input.numeric_selection_time())
        df.at[key, "Filament"] = input.numeric_selection_filament()
        qty_df.set(df)

    @reactive.Effect
    @reactive.event(input.link_comp_more)
    def link_comp_more():
        """
        Display a modal with more instructions

        Reactive Input: link_comp_more
        """
        ui.modal_show(
            ui.modal(
                ui.p("Update Qty to add, remove or change components"),
                ui.p(
                    "To get a better estimate, fill out Print Time and Filament from your slicer"
                ),
                ui.p(
                    'Use the "Only Show Qty>0" to help narrow down which components to slice'
                ),
                ui.p(
                    'Use the "Clear ALL time and filament" button to clear out the default print times and filament usages'
                ),
                title="More Info: Update Components",
                easy_close=True,
            )
        )

    @reactive.Effect
    @reactive.event(input.button_clear_time_filament)
    def button_clear_time_filament():
        """
        Clear all PrintTime and Filament values

        Reactive Input: button_clear_time_fimament
        Reactive Ouputs: qty_df
        Other Inputs: qty_df

        """
        df = qty_df()
        if df is None:
            return
        df = df.assign(PrintTime=NA, Filament=NA)
        qty_df.set(df)

    ########## Step 5 - Review Totals

    @output
    @render.text
    def text_totals_error():
        """
        Show any totataling warnings

        Reactive Input: qty_df
        """
        df = qty_df()
        if df is None:
            return None
        df = df[df["Qty"] > 0]
        if isna(df["Filament"]).any() or isna(df["PrintTime"]).any():  # type: ignore pandas stub is incomplete
            return "* Some values in the total calculation not defined. Check the grid for missing Print Time or Filament values"
        else:
            return ""

    @output
    @render_widget
    def widget_totals():
        """
        Generate a datagrid of totals

        Reactive Inputs: qty_df
        """
        df = qty_df()
        if df is None:
            return dg.DataGrid(DataFrame())
        df = df[df["Qty"] > 0]

        df = df.assign(
            PrintTime=df["Qty"] * df["PrintTime"], Filament=df["Qty"] * df["Filament"]
        )

        pt = df.pivot_table(
            values=["PrintTime", "Filament"],
            index=["Material"],
            aggfunc={"PrintTime": "sum", "Filament": "sum"},
        )
        grid = dg.DataGrid(pt, renderers=dg_renderers(dg))
        grid.layout.height = "80px"
        return grid

    ###### Step 6 - Download datafile

    @session.download(filename="StageTop.csv", media_type="text/csv")
    def button_download_data():
        """
        Reactive Inputs: qty_df
        """
        df = qty_df()
        if df is None:
            return
        df = df.copy()

        df["PrintTime"].apply(format_minutes)
        yield df.to_csv()

    # @output(suspend_when_hidden=False)
    # @render.text
    # def grid_has_selections():
    #    selections = reactive_read(datagrid, "selections")
    #
    #    if (
    #        selections
    #        and len(selections) == 1
    #        and selections[0]["r1"] == selections[0]["r2"]
    #    ):
    #        return "yes"
    #    return "no"


app = App(app_ui, server, debug=False)
