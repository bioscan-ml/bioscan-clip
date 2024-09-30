import os
import csv
import math
import json
import yaml
import numpy as np
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
    flags = {"image": False, "dna": False, "language": False}
    for alignment in ["image", "dna", "language"]:
        if "load_ckpt" in config["model_config"] and config["model_config"]["load_ckpt"] is False:
            alignment_str += "\\myxmark & "
        elif alignment in config["model_config"]:
            alignment_str += "\\checkmark & "
            flags[alignment] = True
        else:
            alignment_str += "\\myxmark & "
        
    return alignment_str, (flags["image"], flags["dna"], flags["language"])


def get_result(csv, header, corr, alignment_type, macro=False):
    
    def compute_HM(seen, unseen):
        if type(seen) == str: seen = float(seen)
        if type(unseen) == str: unseen = float(unseen)

        if seen == 0 or unseen == 0:
            return -2
        else:
            return 2 / (1 / seen + 1 / unseen)

    header_dict = {
        "Order": [10, 14],
        "Family": [11, 15],
        "Genus": [12, 16],
        "Species": [13, 17]
    }

    coor_dict = {
        "dna2dna": {
            (False, False, False):[37, 40],
            (True, True, True):[37, 40],
            (True, True, False):[25, 28],
            (True, False, True):[None, None],
        },
        "img2img": {
            (False, False, False):[1, 4],
            (True, True, True):[1, 4],
            (True, True, False):[1, 4],
            (True, False, True):[1, 4],
        },
        "img2dna": {
            (False, False, False):[7, 10],
            (True, True, True):[7, 10],
            (True, True, False):[7, 10],
            (True, False, True):[None, None],
        }
    }

    row = coor_dict[corr][alignment_type][1 if macro else 0]
    column_list = header_dict[header]

    if row is None:
        return -1, -1, -1
    
    seen = round(float(csv[row][column_list[0]]) * 100, 1)
    unseen = round(float(csv[row][column_list[1]]) * 100, 1)
    harmonic = compute_HM(seen, unseen)

    return seen, unseen, harmonic


def get_results(folder_list, idx, header, corr, macro=False, last=False):
    seen_list = []; unseen_list = []; harmonic_list = []
    return_string = ""

    for folder in folder_list:
        with open(f"{folder}/logs/results.csv", mode='r') as csvfile:
            readCSV = list(csv.reader(csvfile, delimiter=','))
        config = yaml.load(open(f"{folder}/.hydra/config.yaml", "r"), Loader=yaml.FullLoader)
        _, alignment_type = get_alignment(config)
        
        seen, unseen, harmonic = get_result(readCSV, header, corr, alignment_type, macro=macro)
        seen_list.append(seen); unseen_list.append(unseen); harmonic_list.append(harmonic)


    for num_idx, num_list in enumerate([seen_list, unseen_list, harmonic_list]):
        index_max_lst = np.argwhere(num_list == np.max(num_list)).flatten().tolist()
        if idx in index_max_lst:
            return_string += "\\best{%.1f} " % num_list[idx]
        else:

            max_val = np.max(num_list)
            masked_array = np.ma.masked_array(num_list, num_list == max_val)
            index_second_lst = np.argwhere(masked_array == np.max(masked_array)).flatten().tolist()

            if len(index_max_lst) == 1 and len(masked_array) > 0 and idx in index_second_lst:
                return_string += "\\second{%.1f} " % masked_array[index_second_lst]
                
            else:
                if num_list[idx] == -1:
                    return_string += "--- "
                elif num_idx == 2 and num_list[idx] == float(-2):
                    return_string += "{%.1f} " % 0
                else:
                    return_string += "%.1f " % num_list[idx]
        
        if num_idx == 2 and last:
            return_string += "\\\\ \n"
        else:
            return_string += "& "

    return return_string


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
        latex_strings += " rrr rrr rrr rrr rrr rrr@{}}\n"

        latex_strings += "\\toprule\n"
        # write first row of the header
        if args.metric == "both":

            column_starter = 2
            latex_strings += "& "
            if dataset:
                latex_strings += "& "
                column_starter += 1
            if alignment:
                latex_strings += "& & & "
                column_starter += 3
            latex_strings += "\\multicolumn{9}{c}{Micro top-1 accuracy} & \\multicolumn{9}{c}{Macro top-1 accuracy} \\\\ \n"
            latex_strings += "\\cmidrule(l){%d-%d} \\cmidrule(l){%d-%d}\n" % (column_starter, column_starter+8, column_starter+9, column_starter+17)

        # write second row of the header
        column_starter = 2
        latex_strings += "& "
        if dataset:
            latex_strings += "& "
            column_starter += 1
        if alignment:
            latex_strings += "\\multicolumn{3}{c}{Aligned embeddings} & "
            column_starter += 3
        latex_strings += "\\multicolumn{3}{c}{DNA to DNA} & \\multicolumn{3}{c}{Image to Image} & \\multicolumn{3}{c}{Image to DNA} "
        if args.metric == "both":
            latex_strings += "& \\multicolumn{3}{c}{DNA to DNA} & \\multicolumn{3}{c}{Image to Image} & \\multicolumn{3}{c}{Image to DNA}"
        latex_strings += "\\\\ \n"

        if alignment:
            latex_strings += "\\cmidrule(){%d-%d} " % (column_starter - 3, column_starter - 1)
        latex_strings += "\\cmidrule(l){%d-%d} \\cmidrule(l){%d-%d} \\cmidrule(l){%d-%d}" % \
            (column_starter, column_starter + 2, column_starter + 3, column_starter + 5, column_starter + 6, column_starter + 8)
        if args.metric == "both":
            latex_strings += " \\cmidrule(l){%d-%d} \\cmidrule(l){%d-%d} \\cmidrule(l){%d-%d}" % \
            (column_starter + 9, column_starter + 11, column_starter + 12, column_starter + 14, column_starter + 15, column_starter + 17)
        latex_strings += "\n"

        # write third row of the header
        latex_strings += "Taxon & "
        if dataset:
            latex_strings += "Trained on & "
        if alignment:
            latex_strings += "Img & DNA & Txt & "
        latex_strings += "~~Seen & Unseen & H.M. & ~~Seen & Unseen & H.M. & ~~Seen & Unseen & H.M."
        if args.metric == "both":
            latex_strings += " & ~~Seen & Unseen & H.M. & ~~Seen & Unseen & H.M. & ~~Seen & Unseen & H.M."
        latex_strings += "\\\\ \n"

    # write the content
    latex_strings += "\\midrule\n"

    for header in ["Order", "Family", "Genus", "Species"]:
        for idx, dir in enumerate(args.result_folder):
            if idx == 0:
                latex_strings += f"{header} & "
            else:
                latex_strings += " & "

            config = yaml.load(open(f"{dir}/.hydra/config.yaml", "r"), Loader=yaml.FullLoader)

            if dataset:
                latex_strings += get_dataset(config)
            alignment_string, _ = get_alignment(config)
            if alignment:
                latex_strings += alignment_string

            if args.metric == "both":        
                for macro in [False, True]:
                    for corr in ["dna2dna", "img2img", "img2dna"]:
                        latex_strings += get_results(
                            args.result_folder, idx, header, corr, macro=macro, 
                            last=True if corr == "img2dna" and macro is True else False)
            else:
                macro = False if args.metric == "micro" else True
                for corr in ["dna2dna", "img2img", "img2dna"]:
                    latex_strings += get_results(
                        args.result_folder, idx, header, corr, macro=macro, 
                        last=True if corr == "img2dna" else False)

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
    parser.add_argument("--result_folder", type=str, nargs='+', 
                        default=["outputs/2024-09-16/19-20-19", 
                                 "outputs/2024-09-17/06-08-54", 
                                 "outputs/2024-09-16/13-24-08",
                                 "outputs/2024-09-15/17-23-29",])
    # parser.add_argument("--result_folder", type=str, nargs='+')
    parser.add_argument("--full_table", action="store_true",
                        help="Write the full table, including the header and footer")
    parser.add_argument("--no_dataset", action="store_true", help="table does not contain dataset column")
    parser.add_argument("--no_alignment", action="store_true", help="table does not contain alignment column")
    parser.add_argument("--metric", type=str, default="both")

    args = parser.parse_args()

    assert args.metric in ["both", "micro", "macro"], "Invalid metric"

    main(args)