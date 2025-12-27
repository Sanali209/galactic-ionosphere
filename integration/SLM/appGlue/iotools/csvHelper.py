def get_filepaths_from_csv(csvfilepath, column_namber=0, spliter=';', ifnore_first_row=True, encoding='utf8'):
    filepaths = []
    with open(csvfilepath, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if ifnore_first_row:
            lines.pop(0)
        for line in lines:
            line = line.replace('\n', '')
            columns = line.split(spliter)
            filepaths.append(columns[column_namber])
    return filepaths


def get_filepaths_tags_from_csv(csvfilepath, column_namber=0, tags_column=1, spliter=';', ifnore_first_row=True,
                                encoding='utf8'):
    result = []
    with open(csvfilepath, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if ifnore_first_row:
            lines.pop(0)
        for line in lines:
            line = line.replace('\n', '')
            columns = line.split(spliter)
            result.append((columns[column_namber], columns[tags_column]))
    return result
