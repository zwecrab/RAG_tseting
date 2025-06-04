def table_to_text(table):
    """Convert a pandas DataFrame table to a string representation."""
    if table.empty:
        return "Empty table"
    return table.to_string(index=False)