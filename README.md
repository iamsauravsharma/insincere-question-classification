# INSINCERE_QUESTION_CLASSIFICATION

This is a final-year project done by a group of Bishal Gaire, Bishal Rijal, Dilip Gautam and Saurav Sharma. This project classifies whether a question is sincere or insincere using deep learning. It uses [dataset][dataset_link] provided by Kaggle.

**Project Info:**

| License | LoC |
| :---: | :---: |
| [![License: MIT][license_badge]][license_link] | [![LoC][loc_badge]][loc_link] |

**Install**

To install a project Poetry should be installed for package management. You can learn about poetry by visiting their [Github][github_link] or [Website][website_link].

**Running Locally**

This project is built using Google Colab free service so to run locally initially kaggle.json file is required. The steps for installing and authenticating a kaggle CLI can be found [here][kaggle_link].

All Jupyter Notebook files have two initial cells which are used for uploading a kaggle.json file on Google Colab so those initial two cells should be ignored while running locally.

Also, you are not required to run cells that installs a package using the pip command over Google Collab where packages are not preinstalled. All packages can be preinstalled using a poetry locally.

Similarly, a Python file is also present which is built from a jupyter notebook by removing unnecessary code cells from it but kaggle dataset needs to be downloaded early

You need trained model file and tokenizer file to run a Flask app or which can be generated running a Python file named `train_and_save_model.py` for running and training ANN and saving the best model. Initially models folder needs to be created otherwise it may fail to save a model altogether

**Running Over Google Colab**

To run Jupyter Notebook over Google Colab you need to have a kaggle.json file locally. While using Google Collab all cells are required.

To open the notebook in Google Colab you can replace out https://github.com address with https://colab.research.google.com/github or simply install [chrome extension][chrome_link] or [firefox extension][firefox_link].

You can find out more information about opening a GitHub link in Google Collab and other functionality [here][colab_github_demo_link]

**Other Models**

Other models which are tested are also present in different-models branch if you need to see models then you can see them in that branch and run as required. Only the Jupyter Notebook is available for those models.

**Images**

Images such as word cloud, bar and other diagrams are present in the images directory to visualize data length & and distribution.

**Research Paper**

We have also published our research paper on the project if you need to read our research paper and learn about the project then you can read out [research paper](research_paper.pdf) present in this repo.

[dataset_link]: https://www.kaggle.com/c/quora-insincere-questions-classification/data

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
