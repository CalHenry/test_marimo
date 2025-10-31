import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import polars as pl
    from taxonomy_of_messy_data.generate_lvl1_data import generate_messy_dataset
    return Path, generate_messy_dataset, mo, pl


@app.cell
def _(Path, generate_messy_dataset, pl):
    # Read or generate data
    csv_path = Path("data/level_1/messy_dataset.csv")

    if csv_path.exists():
        df = pl.read_csv(csv_path)
    else:
        n_rows = 50
        df = generate_messy_dataset(n_rows, messiness_report=False)
        df.write_csv(csv_path)
    return (df,)


@app.cell
def _(df):
    df_copy = df.clone()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Exploration and to-do-list

    Marimo has a nice ui table to explore the data with summary statistics:
    - shape information (50 rows, 15 columns)
    - variable type (2 int64, 13 str)
    - number of unique for each var
    - graph for numeric variables

    This saves us a few lines of code, but it's always good to have a few lines that check those values.
    """
    )
    return


@app.cell
def _(df):
    # shape
    print(f"Shape: {df.shape}")
    print(f"Schema: {df.schema}")

    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    More informations about the dataframe:
    - no missing (null) values encoded 'None'. Can be placeholders
    - age has impossible values (min, max)
    - inconsistencies in other variables
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To-do-list:

    - to numeric: 'amount', 'revenue', 'commission_rate', 'zip_code'
    - to date: 'transaction_date'
    - remove outliers: 'age'
    - missing value encoding: 'email', 'phone_number', 'city', 'zip_code' and 'department'
    - correct inconsistencies:
        - 'amount': at least 2 formats are present
        - 'date': several encodings (Y/m/d, Y-m-d, d-m-Y, [Oct 09, 2024], maybe more)
        - 'revenue': different currencies and formats ($, €, £, no currency)
        - 'commission_rate': different formats for the %
        - 'is_active': different formats for the boolean var (False, True, F, T, 0, 1, N)
        - 'company_name': upper case and lower case not consistent
        - 'customer_name': whitespaces not consistent
        - 'street_address': names and abbreviations (S., Street, ...)
        - 'job_title': abbreviations (Doctor, Dr., ...)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Convert to numeric

    **'amount'** variable:
    ```python
    df.cast({'amount': pl.Float32})
    ```
    Returns the following error:

    >InvalidOperationError
    >
    >    conversion from `str` to `f32` failed in column 'amount' for 1 out of 50 values: ["8.330,71"]

    We have a European-style number. To convert it to a standard number, we have to remove the thousand delimiter (.), and replace the comma with a period.
    We also have standard form, so we have to use conditions to modify the right numbers.
    """
    )
    return


@app.cell
def _(pl):
    # from european number format to standart number format
    fix_number_format = (
        pl.when(pl.col("amount").str.contains(","))  # European format
        .then(
            pl.col("amount")
            .str.replace_all(r"\.", "")  # Remove periods
            .str.replace(",", ".")
        )
        .otherwise(pl.col("amount"))  # Standard format
        .cast(pl.Float64)
        .round(3)
    )
    return (fix_number_format,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Symbols

    The 'revenue' variable sometimes has a currency symbol, sometimes not.    
    For analytics, it's much better to have actual numbers.   
    We can add the currency in the variable name.    
    Let's remove the symbols and convert the values to the same currency.    

    Let's take fixed values to convert the currencies:    
    1€ = 1.11$    
    1£ = 1.34$
    """
    )
    return


@app.cell
def _(df, pl):
    remove_currency_symbol = (
        pl.col("revenue")
        .str.replace_all(
            r"[$€£,(USD)]", ""
        )  # remove any occurences of "$", "€", "£", "," or "USD"
        .str.strip_chars()  # found whitespaces so we remove them
        .cast(pl.Float64)
        .alias("revenue_clean")
    )

    convert_currency = (
        pl.when(pl.col("revenue").str.contains("€"))
        .then(pl.col("revenue_clean") * 1.11)
        .when(pl.col("revenue").str.contains("£"))
        .then(pl.col("revenue_clean") * 1.34)
        .otherwise(pl.col("revenue_clean"))  # keep at is when already $
        .alias("revenue_usd")
    )


    def fix_currency(df):
        return (
            df.with_columns(remove_currency_symbol)
            .with_columns(convert_currency)
            .with_columns(pl.col("revenue_usd").round().alias("revenue_usd"))
        )


    _df = fix_currency(df)
    return


app._unparsable_cell(
    r"""
    ### Percentages

    The variable **'commission_rate'** is a percentage.  
    We can see at least 3 different expressions (12%, 0.12, 12).  
    Let's standardize it to the default **0.12**
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Percentages

    The variable **'commission_rate'** is a percentage.  
    We can see at least 3 different expressions (12%, 0.12, 12).  
    Let's standardize it to the default **0.12**
    """
    )
    return


@app.cell
def _(pl):
    fix_percentage = (
        pl.col("commission_rate")
        .str.replace("%", "")  # remove %
        .cast(pl.Float64)  # convert to numeric
        .pipe(  # convert to actual percentages
            lambda x: pl.when(x > 1).then(x / 100).otherwise(x)
        )
    )
    return (fix_percentage,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Dates

    A commonly used date format is YYYY-MM-DD.  
    First, we have to find all the date formats that we have.  

    Looking at the dates, we can see more than 3 different formats.  
    To find all the wrong date formats, we can use the error from the function used to convert to the right format. All the formats that fail are wrong, and we can filter them out.   
    Polars can help us here. The function has an argument 'strict=False', that replaces with 'null' all the values that the function couldn't convert.  
    Sometimes the error message gives enough context to directly find the problematic values, so it's good to start with that and write more code only if necessary. (For this case, the error message would be given by this line:  
    ```python
    df.with_columns(pl.col("transaction_date").str.to_date("%Y-%m-%d"))")
    """
    )
    return


@app.cell
def _(df, pl):
    invalid_dates = (
        df.filter(
            pl.col("transaction_date")
            .str.to_date("%Y-%m-%d", strict=False)
            .is_null()
        )
        .select("transaction_date")
        .to_series()
        .to_list()
    )

    print(f"invalid formats: {invalid_dates}")

    # we have to translate this list of dates into a list of formats

    # polars expression
    date_formats = [
        "%d-%m-%Y",
        "%b %d, %Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ]


    fix_date_format = pl.coalesce(
        *[
            pl.col("transaction_date").str.to_date(fmt, strict=False)
            for fmt in date_formats
        ]
    )
    return (fix_date_format,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Breakdown of the expression 'fix_date_format'**:

    - each iteration of the list comprehension creates a new variable with:
      - if it matches the format: a formatted date
      - otherwise: null

    --> we have __*n*__ new variables with the formatted date and nulls. We want to compress those into a single variable, so we use **polars's coalesce** function.

    **pl.coalesce**: folds columns left to right and takes the first non-null value.

    - by taking the first non-null value for each column, we end up with a complete variable unless there are only nulls (if a date didn't match any format)
    ```python
    ┌────────────┬────────────┬────────────┬──────────────┐
    │ col1       │ col2       │ col3       │ coalesce_col │
    │ ---        │ ---        │ ---        │ ---          │
    │ date       │ date       │ date       │ date         │
    ╞════════════╪════════════╪════════════╪══════════════╡
    │ null       │ null       │ 2024-09-22 │ 2024-09-22   │
    │ null       │ null       │ 2024-12-16 │ 2024-12-16   │
    │ null       │ 2024-10-09 │ null       │ 2024-10-09   │
    │ null       │ null       │ 2024-02-11 │ 2024-02-11   │
    │ 2024-04-16 │ null       │ null       │ 2024-04-16   │
    │ null       │ null       │ 2024-02-25 │ 2024-02-25   │
    │ null       │ null       │ 2024-02-20 │ 2024-02-20   │
    └────────────┴────────────┴────────────┴──────────────┘
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Missing values

    Missing values are data points that are absent in the dataset.  

    It's important to mark them with special indicators that are obvious.  
    Many indicators exist and are valid, like **NA**, **NaN**, **None** in Python and R, **NULL** in SQL.  
    Sometimes the missing values are encoded with **placeholders** like impossible values (-999, 9999).  
    It varies with the source of the data and the software used.  

    Nevertheless, here in Python doing data cleaning, we want to have the same indicator for missing values for all the variables. The goal is to have missing values that are easy to **identify** and easy to **fill**.

    To fix the missing values, we have to:
    - Gather all the placeholders, one variable at a time  
    - Store them in a list  
    - Use the list to replace the placeholders with a valid indicator: **None**


    > Why **None**:
    > 
    > **None** is Python's built-in datatype for missing values. It's automatically recognized by the dataframe libraries and converted to their missing value notation: NaN in **pandas** or null in **polars**.
    """
    )
    return


@app.cell
def _(df, pl):
    # 'email'
    _df = df.filter(~pl.col("email").str.contains("@"))
    email_null_placeholders = _df["email"].to_list()

    # 'phone_number'
    _df = df.filter(
        ~pl.col("phone_number").str.contains(r"(\d{3}-)+")
    )  # !TODO breakdown of the regex
    phone_number_null_placeholders = _df["phone_number"].unique().to_list()
    return email_null_placeholders, phone_number_null_placeholders


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Regex breakdown:
    ```python
    r"(\d{3}-)+"
    ```
    **()**: capture what is inside the parenthesis    
    **\d**: a digit between 0 and 9    
    **{3}**: quantifier: we look for exactly 3 digits    

    **+**: matches the previous token (the capturing group) one or more times

    As a sentence:  
    > We are looking for 3 digits and a hyphen ' - ', one or more times.  

    We can capture '123-456-789'  
    But not '12-456-789' or '123 -456-789' or '123456-789'
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **city**

    We have city names mixed with missing value placeholders.  
    City names have a capital first letter. We can use this pattern to find the placeholders.  

    We use the '~' symbol, which means 'NOT', so it inverts the selection.
    """
    )
    return


@app.cell
def _(df, pl):
    city_null_placeholders = (
        df.filter(~pl.col("city").str.contains(r"^[A-Z]{1}[a-z]"))
        .select("city")
        .unique()
        .to_series()  # before this, it's a dataset, after it's a series that can become a list
        .to_list()
    )

    # This pattern do not match 'None', so we have to include it.
    city_null_placeholders.append("None")

    # 'department' placeholders are present in the other list (like in city)
    return (city_null_placeholders,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Regex breakdown:  
    ```python 
    r"^[A-Z]{1}[a-z]"
    ```

    **^**: start of the string  
    **[A-Z]**: any uppercase letter  
    **{1}**: quantifier: exactly 1 capital letter  
    **[a-z]**: any lowercase letter  

    This pattern matches words that start with an uppercase letter and have lowercase letters after.  
    Matches 'Portland' but not 'portland' or 'New-York'
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**zip_code** values are numeric values""")
    return


@app.cell
def _(df, pl):
    # we use '~' to capture all values that are NOT digits
    zip_code_null_placeholders = (
        df.filter(~pl.col("zip_code").str.contains(r"\d"))
        .select("zip_code")
        .unique()
        .to_series()
        .to_list()
    )  # Regex pattern: \d = any digit
    return (zip_code_null_placeholders,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **age**  

    Age is a numeric variable. The placeholders will be numeric as well.  
    Normally, age ranges from 0 to 100, with values outside this range being impossible. 
    We use this rule to spot and remove the placeholders for the missing values in the age variable.
    """
    )
    return


@app.cell
def _(df, pl):
    impossible_age_nbr = (
        df.select(
            pl.col("age").filter((pl.col("age") < 0) | (pl.col("age") > 100))
        )
        .count()
        .item()
    )
    return (impossible_age_nbr,)


@app.cell
def _(
    city_null_placeholders,
    df,
    email_null_placeholders,
    impossible_age_nbr,
    phone_number_null_placeholders,
    zip_code_null_placeholders,
):
    print(f"email_null_placeholders: {email_null_placeholders}")
    print(f"phone_number_null_placeholders: {phone_number_null_placeholders}")
    print(f"city_null_placeholders: {city_null_placeholders}")
    print(f"department unique values: {df['department'].unique().to_list()}")
    print(f"zip_code_null_placeholders: {zip_code_null_placeholders}")
    print(f"Age missing values: {impossible_age_nbr}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    All together:
    - we regroup all the found null placeholders into a single list and use it to clean the variables
    - we build a function to take care of the null placeholders
    """
    )
    return


@app.cell
def _(
    city_null_placeholders,
    email_null_placeholders,
    phone_number_null_placeholders,
    pl,
    zip_code_null_placeholders,
):
    null_placeholders_list = list(
        set(
            email_null_placeholders
            + city_null_placeholders
            + phone_number_null_placeholders
            + zip_code_null_placeholders
        )
    )

    str_cols_with_nulls = [
        "email",
        "phone_number",
        "city",
        "zip_code",
        "department",
    ]


    def fix_missing_values(
        df, str_cols_with_nulls: str | list, null_placeholders_list: str | list
    ):
        df = df.with_columns(
            # string columns
            [
                pl.when(pl.col(var).is_in(null_placeholders_list))
                .then(None)  # None type, converted to null by polars
                .otherwise(pl.col(var))
                .alias(var)
                for var in str_cols_with_nulls
            ]
        ).with_columns(
            # age column
            pl.when((pl.col("age") < 0) | (pl.col("age") > 100))
            .then(None)
            .otherwise(pl.col("age"))
            .alias("age")
        )
        return df
    return fix_missing_values, null_placeholders_list, str_cols_with_nulls


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **is_active** is a boolean. 
    We can think of this as a membership test: you are either in or out, you are a member: True or False.  
    Several encodings exist, like 0 and 1, T and F, Yes and No.  
    But I strongly advise using True and False.  

    **is_active** has more than 2 unique values.

    To solve this issue, we will use the Polars function 'is_in' that returns a boolean with True or False.  
    We define a list of true values; if the value is in the list, then it returns True, otherwise False.
    """
    )
    return


@app.cell
def _(df, pl):
    # Issue: boolean but more than 2 unique values
    print(f"More than 2 unique values: {df['is_active'].unique().to_list()}")

    true_values = ["yes", "Y", "true", "1", "True"]

    fix_is_active = pl.col("is_active").is_in(true_values)
    return (fix_is_active,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **company_name** 

    The variable has mixed lower and upper case.  

    We will convert all the names to lowercase.
    """
    )
    return


@app.cell
def _(pl):
    fix_company_name = pl.col("company_name").str.to_lowercase()
    return (fix_company_name,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **customer_name** 

    The values don't look messy, but they actually have whitespaces at the beginning, the end, or in between the first and last name of the string.  
    We can fix this with Polars: 
    - strip_chars() to remove leading and trailing whitespaces
    - replace_all() for the other whitespaces (2 whitespaces instead of one)
    """
    )
    return


@app.cell
def _(pl):
    fix_whitespaces_customer_name = (
        pl.col("customer_name").str.strip_chars().str.replace_all("  ", " ")
    )
    return (fix_whitespaces_customer_name,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **street_address** 

    We can either use the full words or only the abbreviations.  
    We will use the abreviations and follow the [USPS standards](https://studyinthestates.dhs.gov/sites/default/files/USPS%20Street%20Abbreviations%20Job%20Aid.pdf).
    """
    )
    return


@app.cell
def _(pl):
    mapping = {
        "boulevard": "BLVD",
        "drive": "DR",
        "street": "ST",
        "road": "RD",
        "avenue": "AVE",
    }
    ["Doctor", "Professor", "Mister", "Miss", "Manager"]

    fix_street_address = (
        pl.col("street_address")
        .str.to_lowercase()
        .str.replace_many(mapping)
        .str.to_titlecase()  # add uppercase to the first letter of each word
        .str.replace_all(r"[[:punct:]]", "")  # remove punctuation
    )
    return (fix_street_address,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---

    ## Pipeline

    We have a piece of code to fix each of the variables. We have expressions and functions.  
    Now we have to use them on the dataframe in a pipeline.  

    Each expression doesn't depend on each other, so the order in which we pass them to Polars has no impact.  
    We use ```pipe()``` to apply the functions to the dataset.  
    We apply the changes in a single sequence and let Polars optimize the order of the changes.
    """
    )
    return


@app.cell
def _(
    fix_company_name,
    fix_date_format,
    fix_is_active,
    fix_missing_values,
    fix_number_format,
    fix_percentage,
    fix_street_address,
    fix_whitespaces_customer_name,
    null_placeholders_list,
    str_cols_with_nulls,
):
    df = df.with_columns(
        [
            fix_number_format,
            fix_percentage,
            fix_date_format,
            fix_is_active,
            fix_company_name,
            fix_whitespaces_customer_name,
            fix_street_address,
        ]
    ).pipe(fix_missing_values, str_cols_with_nulls, null_placeholders_list)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""---""")
    return


if __name__ == "__main__":
    app.run()
