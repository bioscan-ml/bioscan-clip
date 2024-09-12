import os
import csv
import json
import yaml
from argparse import ArgumentParser


def get_dataset(config):
    dataset_dict = {
        "bioscan_1m": "BS-1M",
        "bioscan_5m": "BS-5M",
        "INSECT": "INSECT"
    }

    if "dataset" in config["model_config"]:
        return dataset_dict[config["model_config"]["dataset"]] + " & "
    else:
        return "--- & "


def get_alignment(config):
    alignment_str = ""
    for alignment in ["image", "dna", "language"]:
        if alignment in config["model_config"]:
            alignment_str += "\\checkmark & "
        else:
            alignment_str += "\\myxmark & "
    return alignment_str


def get_results(csv, header, corr, macro=False, last=False):
    def compute_HM(seen, unseen):
        if type(seen) == str: seen = float(seen)
        if type(unseen) == str: unseen = float(unseen)
        return round(2 / (1 / seen + 1 / unseen), 4)

    header_dict = {
        "Order": [10, 14],
        "Family": [11, 15],
        "Genus": [12, 16],
        "Species": [13, 17]
    }

    coor_dict = {
        "dna2dna": [37, 40],
        "img2img": [1, 4],
        "img2dna": [7, 10]
    }

    row = coor_dict[corr][1 if macro else 0]
    column_list = header_dict[header]

    seen = csv[row][column_list[0]]
    unseen = csv[row][column_list[1]]
    harmonic = compute_HM(seen, unseen)

    if last:
        return f"{seen} & {unseen} & {harmonic} \\\\ \n"
    else:
        return f"{seen} & {unseen} & {harmonic} & "


def write_latex_table_header():
    latex_strings = ""

    latex_strings += "\\begin{table}[tb]\n"
    latex_strings += "\\centering\n"
    latex_strings += "\\caption{}\n"
    latex_strings += "\\resizebox{\\textwidth}{!}\n"
    latex_strings += "%\\footnotsize\n"

    return latex_strings


def writw_latex_table_footer():
    latex_strings = ""

    latex_strings += "\\label{tab:results}\n"
    # latex_strings += "\\vsapce{-0mm}\n"
    latex_strings += "\\end{table}\n"

    return latex_strings


def write_latex_content(args, dataset=True, alignment=True):
    latex_strings = ""

    if args.full_table:
        latex_strings += write_latex_table_header()

        latex_strings += "{\n"

        # latex_strings += "\\begin{tabular}{@{}ll ccc rrr rrr rrr@{}}\n"
        # set column number
        latex_strings += "\\begin{tabular}{@{}l"
        if dataset:
            latex_strings += "l"
        if alignment:
            latex_strings += " ccc"
        latex_strings += " rrr rrr rrr@{}}\n"

        # write first row of the header
        latex_strings += "\\toprule\n"
        latex_strings += " & "
        if dataset:
            latex_strings += "& "
        if alignment:
            latex_strings += "\\multicolumn{3}{c}{Aligned embeddings} & "
        latex_strings += "\\multicolumn{3}{c}{DNA to DNA} & \\multicolumn{3}{c}{Image to Image} & \\multicolumn{3}{c}{Image to DNA} \\\\ \n"
        column_starter = 3 if dataset else 2
        latex_strings += "\\cmidrule(){%d-%d} \\cmidrule(l){%d-%d} \\cmidrule(l){%d-%d} " % (
        column_starter, column_starter + 2, column_starter + 3, column_starter + 5, column_starter + 6,
        column_starter + 8)
        if alignment:
            latex_strings += "\\cmidrule(l){%d-%d} " % (column_starter + 9, column_starter + 11)
        latex_strings += "\n"

        # write second row of the header
        latex_strings += "Taxon & "
        if dataset:
            latex_strings += "Trained on & "
        if alignment:
            latex_strings += "Img & DNA & Txt & "
        latex_strings += "~~Seen & Unseen & H.M. & ~~Seen & Unseen & H.M. & ~~Seen & Unseen & H.M. \\\\ \n"

    # write the content
    latex_strings += "\\midrule\n"

    for header in ["Order", "Family", "Genus", "Species"]:
        for idx, dir in enumerate(args.result_folder):
            if idx == 0:
                latex_strings += f"{header} & "
            else:
                latex_strings += " & "

            config = yaml.load(open(f"{dir}/.hydra/config.yaml", "r"), Loader=yaml.FullLoader)
            with open(f"{dir}/logs/results.csv", mode='r') as csvfile:
                readCSV = list(csv.reader(csvfile, delimiter=','))

            if dataset:
                latex_strings += get_dataset(config)
            if alignment:
                latex_strings += get_alignment(config)

            for corr in ["dna2dna", "img2img", "img2dna"]:
                latex_strings += get_results(
                    readCSV, header, corr, macro=args.macro, last=False if corr != "img2dna" else True)

        latex_strings += "\\midrule\n" if header != "Species" else "\\bottomrule\n"

    if args.full_table:
        latex_strings += "\\end{tabular}\n"
        latex_strings += "}\n"

        latex_strings += writw_latex_table_footer()

    return latex_strings


def main(args):
    latex = write_latex_content(args, dataset=not args.no_dataset, alignment=not args.no_alignment)
    print(latex)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--result_folder", type=str, nargs='+', default=["outputs/test_for_latex/2024-08-25/18-18-25", "outputs/test_for_latex/2024-09-08/11-52-40"])
    parser.add_argument("--result_folder", type=str, nargs='+')
    parser.add_argument("--full_table", action="store_true",
                        help="Write the full table, including the header and footer")
    parser.add_argument("--no_dataset", action="store_true", help="table does not contain dataset column")
    parser.add_argument("--no_alignment", action="store_true", help="table does not contain alignment column")
    parser.add_argument("--macro", action="store_true", help="Write macro results")

    args = parser.parse_args()

    main(args)