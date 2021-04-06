import os
from numpy import nan, isnan


class LatexDocument(object):
    """"""

    def __init__(self, path, filename, title):
        """"""
        self.path = path
        self.filename = filename
        self.title = title
        self.latex_filename = self.filename.replace("pdf", "tex")

        self.full_latex_path = self.path + "/" + self.latex_filename
        self.full_path = self.path + "/" + self.filename
        self.f = open(self.full_latex_path, "w")
        self.f.write("\\documentclass{article}\n\n")
        self.f.write("\\begin{document}\n\n")
        self.f.write("\\begin{center}\n")
        self.f.write("{}\n".format(self.title))
        self.f.write("\\end{center}\n")

    def save_latex(self):
        """"""
        self.f.write("\\end{document}\n")
        self.f.close()

        os.chdir(self.path)
        os.system("pdflatex {} {}".format(self.full_latex_path, self.full_path))

    def write_table(
        self, table_array, col_headers, row_headers, col_name=None, row_name=None
    ):
        """"""
        self.f.write("\\begin{center}\n")
        self.f.write(
            "\\begin{tabular}{ l "
            + "{}".format(" ".join(["c" for _ in col_headers]))
            + "} \n"
        )

        col_headers_str = " & ".join(["{}".format(x) for x in col_headers])
        start_col_headers = " " if col_name is None else col_name
        self.f.write(start_col_headers + " & {} \\\\\n".format(col_headers_str))

        if not row_name is None:
            self.f.write(
                row_name + " {} \\\\\n".format(" ".join(["&" for _ in col_headers]))
            )
        for idx, row in enumerate(row_headers):
            line = "{} & ".format(row)
            values_str = " & ".join(
                [
                    " " if isnan(val) else "{:.2f}".format(val)
                    for val in table_array[idx, :]
                ]
            )
            line += values_str
            if idx + 1 < table_array.shape[0]:
                line += " \\\\"
            line += "\n"
            self.f.write(line)

        self.f.write("\\end{tabular}\n")
        self.f.write("\\end{center}\n")


def latex_table81(data_array):
    """"""
    current_path = os.getcwd()
    output_path = current_path + "/tables"
    latex_doc = LatexDocument(
        output_path,
        "table8.1.pdf",
        "Table 8.1 - Internet Useage Data - Complete Data Series",
    )
    latex_doc.write_table(data_array, range(6), range(6), "q", "p")
    latex_doc.save_latex()
    os.chdir(current_path)


def latex_table82(data_array):
    """"""
    current_path = os.getcwd()
    output_path = current_path + "/tables"
    latex_doc = LatexDocument(
        output_path,
        "table8.2.pdf",
        "Table 8.2 - Internet Useage Data - Partial Data Series",
    )
    latex_doc.write_table(data_array, range(6), range(6), "q", "p")
    latex_doc.save_latex()
    os.chdir(current_path)
