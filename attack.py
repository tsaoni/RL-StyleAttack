import pandas as pd
from textattack.loggers import CSVLogger

pd.options.display.max_colwidth = 480 # increase colum width so we can actually read the examples

logger = CSVLogger(color_method='html')

for result in attack_results:
    logger.log_attack_result(result)

from IPython.core.display import display, HTML
display(HTML(logger.df[['original_text', 'perturbed_text']].to_html(escape=False)))