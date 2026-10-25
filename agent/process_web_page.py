"""
This script contains a core function to parse a HTML string (of a web page) into
- all tables (use common HTML tag to detect) with their metadata
- remaining text

Implementation tools:
- Use BeautifulSoup to parse the HTML and extract tables.
- Use trafilatura to extract the remaining text content from the HTML.

Algorithm:
1. Parse HTML by bs4, get a list of all table objects in the page
    1.1. For each table bs4 object, get all
        - siblings bs4 object (may have another table. if so, drop from the above list)
        - corresponding headings preceding the table. (tips: scan the lowest-level (let's say hX) heading as siblings of the table or the table parent's, then from hX, scan reversely and get corresponding h(X-1), ..., h1 tag of the table)
    1.2. Drop the table from the bs4 tree
2. Extract the remaining text from the page based on table-free bs4 object. May convert the bs4 object back to HTML then use trafilatura.
"""

import os, re, glob, json
import trafilatura
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup, Tag


def _heading_level(tag_name: str) -> int | None:
    if not tag_name or len(tag_name) != 2 or tag_name[0] != "h" or not tag_name[1].isdigit():
        return None
    level = int(tag_name[1])
    return level if 1 <= level <= 6 else None


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def parse_web_page(html: str):
    soup = BeautifulSoup(html, "html.parser")

    tables = []
    heading_stack: list[tuple[int, str]] = []

    for node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "table"]):
        if not isinstance(node, Tag):
            continue

        if node.name == "table":
            if node.find_parent("table") is not None:
                continue

            table_headings = [{"level": level, "text": text} for level, text in heading_stack]
            df = pd.read_html(StringIO(str(node)))[0]
            df = clean_up_df(df)
            # avg_words_per_cell = df.map(lambda x: len(str(x).split())).mean().mean() if df.size > 0 else 0
            # if avg_words_per_cell > 50: # treat it as text if the table is too big or dense
            #    continue
            tables.append(
                {
                    # "html": str(node),
                    # "text": _clean_text(node.get_text(" ", strip=True)),
                    "dataframe": df.to_dict(orient="records"),
                    "headings": table_headings,
                    "table_metadata": {
                        "id": node.get("id"),
                        "class": node.get("class"),
                        "caption": _clean_text(node.caption.get_text(" ", strip=True)) if node.caption else "",
                    },
                }
            )
            node.extract()
            continue

        level = _heading_level(node.name)
        if level is None:
            continue

        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, _clean_text(node.get_text(" ", strip=True))))

    remaining_html = str(soup)
    remaining_text = trafilatura.extract(remaining_html, output_format="markdown", include_comments=False)
    if not remaining_text:
        remaining_text = _clean_text(soup.get_text(" ", strip=True))

    return {
        "tables": tables,
        "text": remaining_text,
    }


def clean_up_df(df):
    # drop rows and columns with all missing values
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    # convert columns to appropriate/best data types
    df = df.convert_dtypes()
    
    # Try to infer and convert numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass  # Keep as string if conversion fails

    # Remove header level if it only has one value
    while isinstance(df.columns, pd.MultiIndex):
        highest_level = [_ for _ in df.columns.levels[0] if not re.match(r'Unnamed: \d+', str(_))]
        if len(highest_level) <= 1:
            df.columns = df.columns.droplevel(0)
        else:
            break

    # NOTE: the following command converts multi-level column name to single-level column names
    # to simplify the processing with Markdown tables
    if isinstance(df.columns, pd.MultiIndex):
        fn = lambda cols: ' -- '.join(deduplicate_multiindex_column_names(cols))
        df.columns = df.columns.map(fn)

    return df


def deduplicate_multiindex_column_names(cols):
    shortlist = []
    for i, c in enumerate(cols):
        if not re.match(r'Unnamed: \d+', str(c)) or i == len(cols) - 1:
            if c not in shortlist:
                shortlist.append(c)
    return shortlist


def test():
    filepath = 'results/webcontent/html/Top-10-Most-Viewed-Music-Videos-on-YouTube--2025-Edition-.html'
    for filepath in sorted(glob.glob("results/webcontent/html/*.html")):
        if os.path.isfile(filepath):
            html = open(filepath, 'r').read()
            print("\n\n", "="*20, sep='')
            print(filepath)
            result = parse_web_page(html)
            print("\n```python\n" + json.dumps(result, indent=2) + "\n```")


if __name__ == "__main__":
    test()
    # Checked. Good!


"""
URL lists
https://data.un.org/en/reg/g15.html
https://www.cardrates.com/advice/credit-card-processors-market-share/
https://www.forbes.com/advisor/investing/fed-funds-rate-history/
https://en.wikipedia.org/wiki/List_of_most-viewed_YouTube_videos
https://en.wikipedia.org/wiki/List_of_Nobel_laureates_in_Physics
https://rebelmusicz.com/most-viewed-music-videos-on-youtube/
https://www.boxofficemojo.com/chart/top_lifetime_gross/
https://en.wikipedia.org/wiki/Venezuela
https://en.wikipedia.org/wiki/List_of_The_Simpsons_episodes
https://nbastats.net/
"""
