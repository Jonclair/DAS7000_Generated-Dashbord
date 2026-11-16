import io
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="EV Adoption in Washington", layout="wide")

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data["Legislative District"] = data["Legislative District"].astype("Int64")
    return data

df = load_data("ev_clean_wa.csv")

st.title("Electric Vehicle Adoption in Washington State")

st.info(
    "This app explores the cleaned Washington Electric Vehicle Population dataset to help a "
    "regional charging-network planner understand how many electric vehicles are registered, "
    "what types and makes they are, and where they are located. It describes vehicles currently "
    "on the register by model year."
)

st.header("Data Overview")

st.subheader("Data Preview")
rows_to_show = st.selectbox(
    "Number of rows to display",
    [5, 10, 20, 50, 100]
)
st.dataframe(df.head(rows_to_show))

st.subheader("Dataset Structure")
col1, col2 = st.columns(2)
col1.metric("Number of Rows", f"{df.shape[0]:,}")
col2.metric("Number of Columns", df.shape[1])

st.write(df.columns.tolist())

categorical_variables = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
numerical_variables = df.select_dtypes(include="number").columns.tolist()

cat_col, num_col = st.columns(2)
with cat_col:
    st.write(f"**Categorical variables ({len(categorical_variables)})**")
    st.write(categorical_variables)
with num_col:
    st.write(f"**Numerical variables ({len(numerical_variables)})**")
    st.write(numerical_variables)

identifier_columns = ["Postal Code", "Legislative District", "GEOID 2020", "DOL Vehicle ID"]
st.caption(
    "Postal Code, Legislative District, GEOID 2020 and DOL Vehicle ID are stored as numbers "
    "but are identifiers (labels), not measurements."
)

with st.expander("View Dataset Information"):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()
    st.text(info_text)

st.subheader("Statistical Summary")
numeric_for_summary = [c for c in numerical_variables if c not in identifier_columns]
st.dataframe(df[numeric_for_summary].describe().T)
st.caption(
    "Electric Range is blank for vehicles whose range was never recorded, so its summary uses "
    "only the vehicles that have a recorded range."
)

st.subheader("Population Summary")

total_vehicles = len(df)
bev_share = (df["Vehicle Type"] == "BEV").mean() * 100

make_counts = df["Make"].value_counts()
top_make = make_counts.index[0]
top_make_share = make_counts.iloc[0] / total_vehicles * 100

county_counts = df["County"].value_counts()
top_county = county_counts.index[0]
top_county_share = county_counts.iloc[0] / total_vehicles * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total EVs registered", f"{total_vehicles:,}")
col2.metric("Battery electric (BEV) share", f"{bev_share:.1f}%")
col3.metric(f"Top make: {top_make.title()}", f"{top_make_share:.1f}%")
col4.metric(f"Top county: {top_county}", f"{top_county_share:.1f}%")

st.caption(
    f"Model years {df['Model Year'].min()} to {df['Model Year'].max()}. "
    "Model year is not the same as registration year."
)

st.header("Data Cleaning Process")

tab_summary, tab_steps, tab_log, tab_checks, tab_remaining = st.tabs(
    ["Summary", "Step by step", "Decision log", "Validation checks", "Remaining issues"]
)

with tab_summary:
    st.write(
        "The raw file was cleaned in a Python notebook before it was used in this app. "
        "The table compares the raw data with the cleaned data shown here."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}", "-789 vs raw", delta_color="off")
    c2.metric("Columns", df.shape[1], "+9 vs raw", delta_color="off")
    c3.metric("Electric Range zeros", int((df["Electric Range"] == 0).sum()),
              "-197,085 vs raw", delta_color="off")

    id_cols = ["Postal Code", "Legislative District", "GEOID 2020"]
    city_case_diff = (
        df["City"].nunique()
        - df["City"].astype("string").str.strip().str.upper().nunique()
    )
    before_after = pd.DataFrame({
        "Measure": [
            "Rows",
            "Rows outside Washington",
            "Rows with Electric Range = 0",
            "Rows with Electric Range missing",
            "Missing Legislative District",
            "Missing coordinates",
            "City names differing only by capitalisation",
            "Identifier columns stored as decimals",
        ],
        "Before cleaning": ["300,142", "789", "197,085", "35", "790", "24", "2", "3"],
        "After cleaning": [
            f"{len(df):,}",
            f"{int((df['State'] != 'WA').sum()):,}",
            f"{int((df['Electric Range'] == 0).sum()):,}",
            f"{int(df['Electric Range'].isna().sum()):,}",
            f"{int(df['Legislative District'].isna().sum()):,}",
            f"{int(df['Latitude'].isna().sum()):,}",
            str(city_case_diff),
            str(sum(df[c].dtype == "float64" for c in id_cols)),
        ],
    })
    st.table(before_after.set_index("Measure"))
    st.caption(
        "Electric Range zeros were converted to missing values, so the missing count "
        "rises as the zeros fall. No range values were estimated."
    )

with tab_steps:
    st.write("Each step below was applied in order. Row counts are from the notebook.")
    steps = pd.DataFrame(
        [
            ("1. Rename columns",
             "Renamed VIN (1-10) to VIN Prefix, the long CAFV column to CAFV Eligibility, "
             "and 2020 GEOID to GEOID 2020.",
             "No values changed.",
             "Shorter names are easier to use in code and dashboards."),
            ("2. Standardise text",
             "Removed stray spaces. Put Make and Model in upper case and City and County "
             "in Title Case.",
             "Unique city names fell from 934 to 932.",
             "The same place or make must not be counted twice."),
            ("3. Electric Range",
             "Converted zeros to missing values and added a Range Recorded flag.",
             "197,085 zeros became missing. 103,022 vehicles had a recorded range.",
             "Zero means the range was never recorded. Averaging zeros would understate range."),
            ("4. Identifier columns",
             "Converted Postal Code, Legislative District, GEOID 2020 and DOL Vehicle ID "
             "to whole numbers or text.",
             "Three columns are no longer stored as decimals.",
             "They are labels, not quantities, and should never be averaged."),
            ("5. Flag out-of-state rows",
             "Added an Is WA flag and checked the rows missing county, city and utility.",
             "789 rows were outside Washington, including all 15 rows missing "
             "county, city and utility.",
             "Lets out-of-state rows be handled separately."),
            ("6. Trend window",
             "Added a Trend Window flag for model years 2012 to 2026.",
             "298,531 rows inside the window and 1,611 outside it.",
             "Sparse early years and the partial 2027 year distort trends."),
            ("7. Parse coordinates",
             "Extracted Longitude and Latitude from the Vehicle Location text and "
             "checked them against Washington's boundaries.",
             "All Washington coordinates were inside the boundary. "
             "9 Washington rows have no coordinates.",
             "Needed for the map. The check catches wrong locations."),
            ("8. Split utilities",
             "Split Electric Utility on | and || into a list. Created Utility Count "
             "and Primary Utility (the first listed).",
             "224,514 vehicles list more than one utility.",
             "Counting the raw text would treat each combination as a separate utility."),
            ("9. Short labels",
             "Created Vehicle Type (BEV or PHEV) and CAFV Status (Eligible, "
             "Not eligible, Unknown).",
             "No missing labels were produced.",
             "Short labels read better in charts and filters."),
            ("10. Washington only",
             "Kept Washington rows for the main analysis and set the others aside.",
             "300,142 rows became 299,353.",
             "Out-of-state rows are 0.26% of the data and mostly lack location details."),
            ("11. Validation checks",
             "Ran automatic checks on the cleaned table.",
             "All checks passed (see the Validation checks tab).",
             "Catches cleaning errors that would otherwise go unnoticed."),
            ("12. Save cleaned data",
             "Saved the cleaned table as ev_clean_wa.csv.",
             "299,353 rows and 25 columns.",
             "One clean source for the dashboard and Tableau."),
        ],
        columns=["Step", "What was done", "Result", "Why"],
    )
    st.table(steps.set_index("Step"))

with tab_log:
    st.write("Every decision is recorded as what was found, what was changed and why.")
    decision_log = pd.DataFrame(
        [
            ("Electric Range",
             "65.7% of rows had range = 0, all BEVs. The zeros start abruptly in model year 2021.",
             "Converted 0 to missing and added a Range Recorded flag. "
             "No range trend beyond 2020.",
             "Zeros mean not recorded. Keeping them would understate range and "
             "suggest a false decline."),
            ("Geography",
             "789 rows were outside Washington. The 15 rows missing county, city and "
             "utility were among them.",
             "Main analysis uses Washington only. All rows kept for sensitivity testing.",
             "Out-of-state rows are 0.26% of the data and lack county and utility details."),
            ("Model year",
             "Very few vehicles before 2012. 2027 has only 1,040 vehicles and looks incomplete.",
             "Added a Trend Window flag (2012 to 2026) for trend charts.",
             "Sparse or partial years would distort growth and share trends."),
            ("Electric Utility",
             "75% of rows list several utilities, separated by | or ||.",
             "Split into a list. The main analysis uses the first listed utility.",
             "Avoids double-counting vehicles. The other counting rule is tested "
             "in sensitivity analysis."),
            ("Vehicle Location",
             "Stored as text in the form POINT (lon lat).",
             "Parsed into numeric Longitude and Latitude and checked against "
             "Washington's boundaries.",
             "Needed for the map."),
            ("Identifier columns",
             "Postal Code, Legislative District and GEOID were stored as decimals.",
             "Converted to whole numbers or text.",
             "They are labels, not quantities."),
            ("City names",
             "Two cities had inconsistent capitalisation, such as McCleary and Mccleary.",
             "Standardised the capitalisation of City and County.",
             "Prevents the same place being counted as two."),
            ("PHEV range outliers",
             "The IQR rule flagged 1,291 PHEV ranges. 1,289 of them are the BMW i3 "
             "(72 to 153 miles).",
             "Kept them.",
             "They come from one real model with a longer electric range than "
             "typical PHEVs, so they are genuine values."),
            ("Duplicates",
             "No duplicate rows or vehicle IDs.",
             "No change.",
             "Nothing to remove."),
        ],
        columns=["Area", "What I found", "What I changed", "Why"],
    )
    st.table(decision_log.set_index("Area"))

with tab_checks:
    st.write("These checks run on the data loaded into this app each time it opens.")
    checks = [
        ("No duplicate vehicle IDs", df["DOL Vehicle ID"].is_unique),
        ("No zero Electric Range values remain", bool((df["Electric Range"].dropna() > 0).all())),
        ("Vehicle Type is only BEV or PHEV", bool(df["Vehicle Type"].isin(["BEV", "PHEV"]).all())),
        ("No missing County or Primary Utility",
         bool(df["County"].notna().all() and df["Primary Utility"].notna().all())),
        ("Every vehicle has at least one utility", bool((df["Utility Count"] >= 1).all())),
        ("All rows are in Washington", bool((df["State"] == "WA").all())),
    ]
    checks_df = pd.DataFrame(
        [(name, "Passed" if ok else "Failed") for name, ok in checks],
        columns=["Check", "Result"],
    )
    st.table(checks_df.set_index("Check"))

with tab_remaining:
    st.markdown("**Missing values still in the cleaned data**")
    missing = df.isna().sum()
    missing = missing[missing > 0].rename("Missing values").to_frame()
    missing["% of rows"] = (missing["Missing values"] / len(df) * 100).round(2)
    st.dataframe(missing)

    st.markdown("**Duplicates**")
    st.write(
        f"Duplicate rows: {df.duplicated().sum()}. "
        f"Duplicate vehicle IDs: {df['DOL Vehicle ID'].duplicated().sum()}."
    )

    st.markdown("**Imputation**")
    st.write(
        "No values were imputed. Missing ranges were left missing "
        "because estimating them would invent data."
    )

    st.markdown("**What the user should keep in mind**")
    st.markdown(
        "- Electric Range is missing for about two-thirds of vehicles, including almost all "
        "BEVs from model year 2021 onwards, so range results are not a trend.\n"
        "- Counts are by model year, not by year of registration or sale.\n"
        "- Only Washington is covered.\n"
        "- Where a vehicle lists several utilities, only the first is used."
    )
