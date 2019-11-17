# INSINCERE_QUESTION_CLASSIFICATION

**Status:**

| Travis Build Status | Code style |
| :---: | :---: |
| [![Travis Build Status][build_badge]][build_link] | [![Code style][black_badge]][black_link] |

**Project Info:**

| License | LoC |
| :---: | :---: |
| [![License: MIT][license_badge]][license_link] | [![LoC][loc_badge]][loc_link] |

**Install**

To install a project at first Poetry should be installed for a package management. You can learn about poetry by visiting their [Github][github_link] 
or [Website][website_link]. Simply use following command to install poetry

```
curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
```

To start out a shell where all package are installed out you can run out command ```poetry shell``` which installs out all package and spawns shell
over virtual environment. To learn more about poetry you can view [poetry docs][poetry_docs_link].

**Running Locally**

This project are build using google colab free service so to run locally initially kaggle.json file is required to run project locally.
The steps for installing and authenticating a kaggle CLI can be found in [here][kaggle_link].

All jupyter notebook file have two initial cell which are used for uploading a kaggle.json file on google colab so those initial two cells
need not to be executed while running locally and may fail.

As well as you are not required to run all cell which are installing a package using pip command over google colab where a package is not preinstalled
over there and need to be installed. All package are preinstalled using a poetry locally.

Similarly python file is also present which is built form a jupyter notebook by removing out unnecessary code cell form it so it can be easily run locally
without problem in the poetry virtual environment of the project. But there are some changes using it such as kaggle dataset need to be downloaded early so
it can be used by python file. 

You can save and store out a trained model file and tokenizer file over a models file to run a flask app or simply train and save model directly by running
a python file named as train_and_save_model for running & training ANN and saving best model. Initially models folder need to be created out otherwise it may
fail to save a model altogether

You can also easily modify out a file path in a notebook and python file file path for different file and folders according to your need 

**Running Over Google Colab**

To run jupyter notebook over google colab you need to have a kaggle.json file locally to be uploaded over google colab server. While using over google colab
all cells need to be run over.

To open notebook in google colab you can replace out https://github.com address to  https://colab.research.google.com/github or simply install
[chrome extension][chrome_link] or [firefox extension][firefox_link].

You can find out more information about opening a github link in google colab and other functionality [here][colab_github_demo_link]

**Other Models**
Other models which are tested out is present in different-models branch if you need to see out models then you can see in that branch and run as required.
Only jupyter notebook version model is available for those models their is no python file as well as you can load out model at google colab easily from each
notebook file.

**Images**
Images such as word cloud, bar and other diagrams is present over images directory to visualize out data.

**Research Paper**

We have also published out research paper of project if you need to read out research paper and learn about project then you can read out
[research paper](research_paper.pdf) present in this repo.

[build_badge]: https://img.shields.io/travis/com/iamsauravsharma/insincere-question-classification.svg?logo=travis
[build_link]: https://travis-ci.com/iamsauravsharma/insincere-question-classificaton

[black_badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[black_link]: https://github.com/ambv/black

[license_badge]: https://img.shields.io/github/license/iamsauravsharma/insincere-question-classification.svg
[license_link]: LICENSE

[loc_badge]: https://tokei.rs/b1/github/iamsauravsharma/insincere-question-classification
[loc_link]: https://github.com/iamsauravsharma/insincere-question-classification

[github_link]: https://github.com/sdispater/poetry
[website_link]: https://poetry.eustace.io/

[poetry_docs_link]: https://poetry.eustace.io/docs/

[kaggle_link]: https://www.kaggle.com/docs/api#getting-started-installation-&-authentication

[chrome_link]: https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo
[firefox_link]: https://addons.mozilla.org/en-US/firefox/addon/open-in-colab/

[colab_github_demo_link]: https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb