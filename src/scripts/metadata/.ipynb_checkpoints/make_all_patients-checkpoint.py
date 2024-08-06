import pandas as pd
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Build metadata json from tsv')
    # parser.add_argument('--tsv_path', type=str, default='F:\\tmp_pancreatic\\temp_tsv',
    #                     help="Where all raw tsv files are stored")
    parser.add_argument('--tsv_path', type=str, default='F:\\tmp_pancreatic\\temp_tsv\\global\\icd_split',
                        help="Where all raw tsv files are stored")
    parser.add_argument('--library_path', type=str, help="Where to general the all_icd.pkl file. If empty, do not generate.")
    parser.add_argument('--delimiter', type=str, default='\t')
    parser.add_argument('--head', type=int, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    file_paths = []
    for root, subdirs, files in os.walk(args.tsv_path):
        if len(files) > 0 and 'demo.tsv' not in files:
            file_path = os.path.join(root, files[0])
            file_paths.append(file_path)

    # df5 = pd.concat([pd.read_csv(file, sep="\t", engine="pyarrow") for file in file_paths[19:24]])
    # df5 = df5[['PatientICN', 'ICDCode', 'ICDCodeType', 'Date']]
    df0 = pd.concat([pd.read_csv(file, sep="\t", engine="pyarrow") for file in file_paths[0:3]])
    # df1 = pd.concat([pd.read_csv(file, sep="\t", engine="pyarrow") for file in file_paths[3:7]])
    # df2 = pd.concat([pd.read_csv(file, sep="\t", engine="pyarrow") for file in file_paths[7:11]])
    # df3 = pd.concat([pd.read_csv(file, sep="\t", engine="pyarrow") for file in file_paths[11:15]])
    # df4 = pd.concat([pd.read_csv(file, sep="\t", engine="pyarrow") for file in file_paths[15:19]])

    # df_final = pd.concat([df0, df1, df2, df3, df4, df5])
    # final_fname = os.path.join(args.tsv_path, "global\\icd_code_counts_final.tsv")
    # df_final.to_csv(final_fname, sep="\t", index=False)
    final_fname = os.path.join("global/icd_split/icd_code_counts_final_3m.tsv")
    df0.to_csv()

# tsv_path ='F:\\tmp_pancreatic\\temp_tsv'
# file_paths = []
# for root, subdirs, files in os.walk(tsv_path):
#     if len(files) > 0 and 'demo.tsv' not in files:
#         file_path = os.path.join(root, files[0])
#         file_paths.append(file_path)
#
# df5 = pd.concat([pd.read_csv(file, sep="\t", engine="pyarrow") for file in file_paths[19:24]])
# df5 = df5[['PatientICN', 'ICDCode', 'ICDCodeType', 'Date']]
# df0 = pd.concat([pd.read_csv(file, sep="\t", engine="pyarrow") for file in file_paths[0:3]])
# df1 = pd.concat([pd.read_csv(file, sep="\t", engine="pyarrow") for file in file_paths[3:7]])
# df2 = pd.concat([pd.read_csv(file, sep="\t", engine="pyarrow") for file in file_paths[7:11]])
# df3 = pd.concat([pd.read_csv(file, sep="\t", engine="pyarrow") for file in file_paths[11:15]])
# df4 = pd.concat([pd.read_csv(file, sep="\t", engine="pyarrow") for file in file_paths[15:19]])
#
# df_final = pd.concat([df0, df1, df2, df3, df4, df5])
# final_fname = 'F:\\tmp_pancreatic\\temp_tsv\\global\\icd_code_counts_final.tsv'
# df_final.to_csv(final_fname, sep="\t", index=False)
# # demo_fname = 'F:\\tmp_pancreatic\\temp_tsv\\global\\raw\\demo.tsv'
# # demo = pd.read_csv(demo_fname, sep="\t")
# # demo_final = demo[['PatientICN', 'BirthDate', 'DeathDate', 'Gender']]
# #
# # patients_all = pd.merge(demo_final, df_final, how="left", on="PatientICN")
#
# f = {'ICDCode': "nunique"}
# f2 = {'Date': "nunique"}
# panc_preEd_all_summmay = df_final.groupby(by="PatientICN")
# v1 = panc_preEd_all_summmay.agg(f)
# V2 = panc_preEd_all_summmay.agg(f2)
#
# # dplyr::group_by(PatientICN) % > %
# # dplyr::summarise(visits=length(unique(Date)),
# #                  num_of_codes=length(unique(ICDCode))) % > %
# # dplyr::mutate(is_panc=ifelse(PatientICN % in % panc_canr$PatientICN, 1, 0)) % > %
# # dplyr::ungroup()
# #
# # panc_pred_all_summmay_2 < - panc_pred_all % > %
# # dplyr::group_by(PatientICN, Date) % > %
# # dplyr::summarise(num_of_codes=length(unique(ICDCode)))
# #
# # panc_pred_all_eligible < - panc_pred_all_summmay % > %
# # dplyr::filter(visits >= 5)
# #
# # final_fname = 'F:\\tmp_pancreatic\\temp_tsv\\global\\icd_code_counts_final.tsv'
# # # df_final.to_csv(final_fname,sep="\t", index=False)
# # pa_table = pa.Table.from_pandas(df_final)
# # csv.write_csv(pa_table, final_fname)





