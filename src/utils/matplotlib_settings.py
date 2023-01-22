import matplotlib as mpl


# Relevant preamble from thesis template
latex_preamble = [
    r'\usepackage{microtype}',
    r'\usepackage{mathtools}',
    r'\usepackage{fontspec}',
    r'\setsansfont[Ligatures=TeX,Extension=.otf,UprightFont=*-regular,'
        r'BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic,'
        r'Scale=0.8]{texgyreadventor}'
    r'\usepackage{amsfonts}',
]

# DTU colors
dtu_colors = [
    '#2F3EEA',
    '#1FD082',
    '#030F4F',
    '#F6D04D',
    '#FC7634',
    '#F7BBB1',
    '#DADADA',
    '#E83F48',
    '#008835',
    '#79238E',
    '#990000',
    '#000000',
]
dtu_color_names = [
    'blue',
    'bright_green',
    'navy_blue',
    'yellow',
    'orange',
    'pink',
    'grey',
    'red',
    'green',
    'purple',
    'dtu_red',
    'black',
]
dtu_colors_dict = {
    name: color for name, color in zip(dtu_color_names, dtu_colors)
}

# Dimensions of thesis template
line_width = 5.1899 # in inches
text_width = 5.1899 # in inches
text_height = 7.61185 # in inches
col_width = 2.52574 # in inches

default_ratio = 3/4
default_width = text_width
default_height = default_width * default_ratio

def set_matplotlib_defaults(font_size=10, font_scale=1.0, backend='pgf'):
    font_size = font_size*font_scale

    mpl.use(backend)
    mpl.rcParams.update({
        'pgf.texsystem': 'xelatex',
        'font.family': 'sans-serif',
        'font.sans-serif': [],
        'text.usetex': True,
        'pgf.rcfonts': False,
        'pgf.preamble': "\n".join(latex_preamble),
        'figure.figsize': [default_width, default_height],
        'font.size': font_size,
        'figure.titlesize': font_size,
        'axes.titlesize': font_size,
        'figure.labelsize': font_size,#9,
        'axes.labelsize': font_size,#9,
        'legend.title_fontsize': font_size,#9,
        'legend.fontsize': font_size,#9,
        'xtick.labelsize': font_size,#8,
        'ytick.labelsize': font_size,#8,
        'xtick.major.size': 3.5,
        'xtick.minor.size': 2.0,
        'ytick.major.size': 3.5,
        'ytick.minor.size': 2.0,
        'axes.prop_cycle': mpl.cycler(color=dtu_colors),
        'axes.spines.right': False,
        'axes.spines.top': False,
    })

    # Change default colors
    mpl.colors.get_named_colors_mapping()['b'] = dtu_colors_dict['blue']
    mpl.colors.get_named_colors_mapping()['g'] = dtu_colors_dict['green']
    mpl.colors.get_named_colors_mapping()['r'] = dtu_colors_dict['red']
    mpl.colors.get_named_colors_mapping()['c'] = dtu_colors_dict['bright_green']
    mpl.colors.get_named_colors_mapping()['m'] = dtu_colors_dict['purple']
    mpl.colors.get_named_colors_mapping()['y'] = dtu_colors_dict['yellow']

    mpl.colors.get_named_colors_mapping()['blue'] = dtu_colors_dict['blue']
    mpl.colors.get_named_colors_mapping()['lightgreen'] = dtu_colors_dict['bright_green']
    mpl.colors.get_named_colors_mapping()['navy'] = dtu_colors_dict['navy_blue']
    mpl.colors.get_named_colors_mapping()['yellow'] = dtu_colors_dict['yellow']
    mpl.colors.get_named_colors_mapping()['orange'] = dtu_colors_dict['orange']
    mpl.colors.get_named_colors_mapping()['pink'] = dtu_colors_dict['pink']
    mpl.colors.get_named_colors_mapping()['grey'] = dtu_colors_dict['grey']
    mpl.colors.get_named_colors_mapping()['gray'] = dtu_colors_dict['grey']
    mpl.colors.get_named_colors_mapping()['red'] = dtu_colors_dict['red']
    mpl.colors.get_named_colors_mapping()['green'] = dtu_colors_dict['green']
    mpl.colors.get_named_colors_mapping()['purple'] = dtu_colors_dict['purple']