chapterColors = [None]*23
chapterColors[0] = (0,0,0)
chapterColors[1] = (47, 207, 211)
chapterColors[2] = (158, 79, 70)
chapterColors[3] = (246, 158, 220)
chapterColors[4] = (255, 26, 185)
chapterColors[5] = (45, 201, 45)
chapterColors[6] = (0, 77, 230)
chapterColors[7] = (255, 193, 132)
chapterColors[8] = (167, 167, 249)
chapterColors[9] = (160, 12, 196)
chapterColors[10] = (98, 93, 93)
chapterColors[11] = (0, 132, 149)
chapterColors[12] = (114, 150, 86)
chapterColors[13] = (150, 8, 102)
chapterColors[14] = (249, 223, 30)
chapterColors[15] = (23, 206, 149)
chapterColors[16] = (18, 96, 27)
chapterColors[17] = (246, 132, 9)
chapterColors[18] = (0, 0, 123)
chapterColors[19] = (255, 0, 0) 
chapterColors[20] = (183, 201, 75)
chapterColors[21] = (0, 0, 2)
chapterColors[22] = (207, 185,151)

chapterColors.pop(0)
chapterColors = [(r/255,g/255,b/255) for r,g,b, in chapterColors]


DK_ICD10_ICD8_DISEASE_HISTOGRAM = {"DE11":"Non-insulin dependent diabetes mellitus",
    "250":"Non-insulin dependent diabetes mellitus", 
    "DE10":"Insulin-dependent diabetes mellitus", 
    "249":"Insulin-dependent diabetes mellitus", 
    "DR17":"Unspecified jaundice", 
    "785":"Unspecified jaundice",
    "DK85":"Acute pancreatitis", 
    "577":"Acute pancreatitis", 
    "DE78":"Hypercholesterolemia", 
    "279":"Hypercholesterolemia", 
    "DR63":"Weight loss and other food intake problems", 
    "784":"Weight loss and other food intake problems", 
    "DK86":"Other diseases of the pancreas", 
    # "577":"Other diseases of the pancreas", REDUNDATNT IN ICD8
    "DC18":"Malignant neoplasm of colon", 
    "153":"Malignant neoplasm of colon", 
    "DC24":"Malignant neoplasm in other and \nunspecified parts of bile ducts",
    "156":"Malignant neoplasm in other and \nunspecified parts of bile ducts",
    "DE66":"Obesity",
    "277": "Obesity",
    "DK50":"Inflammatory bowel disease",
    "DK51":"Inflammatory bowel disease",
    "DK52":"Inflammatory bowel disease",
    "563": "Inflammatory bowel disease",
    }

US_ICD10_ICD9_DISEASE_HISTOGRAM = {"E11":"Type 2 diabetes mellitus",
    "250":"Type 2 diabetes mellitus", 
    "E10":"Type 1 diabetes mellitus",
    "249":"Type 1 diabetes mellitus", 
    "R17":"Unspecified jaundice",
    "7824":"Unspecified jaundice",
    "K85":"Acute pancreatitis",
    "577":"Acute pancreatitis", 
    "E78":"Hypercholesterolemia",
    "272":"Hypercholesterolemia", 
    "R63":"Weight loss and other food intake problems",
    "783":"Weight loss and other food intake problems", 
    # "K86":"Other diseases of the pancreas",
    # "577":"Other diseases of the pancreas",
    "C18":"Malignant neoplasm of colon",
    "153":"Malignant neoplasm of colon", 
    "C24":"Malignant neoplasm in other and \nunspecified parts of bile ducts",
    "156":"Malignant neoplasm in other and \nunspecified parts of bile ducts",
    "E66":"Obesity",
    "278": "Obesity",
    "K50":"Inflammatory bowel disease",
    "K51":"Inflammatory bowel disease",
    "K52":"Inflammatory bowel disease",
    "558": "Inflammatory bowel disease",
    "556": "Inflammatory bowel disease",
    "555": "Inflammatory bowel disease"
    }

def save_figure_and_subplots(figname, fig, **kwargs):
    """
        figname: name of the file to save
        fig: matplotlib.pyplot.figure object

        This function takes some figure and save its content. In addition it save al; its axes/subfigure
        for offline processing.
        
    """
    if 'format' not in kwargs:
        kwargs['format'] = 'png'
    for ax_num,ax in enumerate(fig.axes):
        extent = ax.get_tightbbox(renderer=fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig("{}_{}.{}".format(figname, ax_num, kwargs['format']), bbox_inches=extent, **kwargs)
    fig.savefig("{}.{}".format(figname, kwargs['format']), bbox_inches='tight', **kwargs)