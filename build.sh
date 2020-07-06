#!/usr/bin/bash -i
conda activate sb_covid
jupyter nbconvert --to notebook --execute SB_Covid.ipynb
jupyter nbconvert --to notebook --execute SantaMariaModel.ipynb
jupyter nbconvert SB_Covid.nbconvert.ipynb --to html --no-input --output index.html
jupyter nbconvert SantaMariaModel.nbconvert.ipynb --to html --no-input --output SM_model_page.html
conda deactivate
sensible-browser index.html