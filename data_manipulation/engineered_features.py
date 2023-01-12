import pandas as pd


def has_cabin_feature(data: pd.DataFrame) -> list:
    return [1 if pd.isna(c) else 0 for c in data['Cabin']]


def family_size_feature(data: pd.DataFrame) -> list:
    return [1 + d.SibSp + d.Parch for d in data.itertuples()]


def title_feature(data: pd.DataFrame) -> list:
    return [_extract_title_from_name(n) for n in data.Name]


def _extract_title_from_name(name: str) -> str:
    first_name = name.split(', ')[1]
    first_name_components = first_name.split(' ')
    title = first_name_components[0]

    return title
