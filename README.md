# Beyond the Stars: Quantifying the Sentiment-Rating Gap in Bangladesh's Mobile Financial Services

A multilingual NLP analysis of Google Play reviews for bKash, Nagad, and Rocket. Star ratings and review text often disagree. This project quantifies that mismatch with a custom **Gap Score** (normalized rating vs. VADER sentiment), classifies reviews as *Inflated*, *Deflated*, or *Consistent*, and surfaces silent complaint themes via LDA topic modeling. 

The pipeline is run twice: once on English-only reviews, once on the full Bangla + English (translated) corpus in order to test whether excluding non-English text changes the conclusions.

## Setting up your project:

- Install python 3.12.x
- Open command prompt and run the following commands in correct order:
  - 'git clone https://github.com/JawwadHossain/K502_project.git
  - cd K502_project
  - 'python -m venv proj_env'
  - 'proj_env\Scripts\activate.bat'
  - 'python -m pip install -r requirements.txt'

## Executing the program

**You can execute the program from either of the three files:**

- notebooks/main_standalone.ipynb
  
  - No custom module dependency
  
  - Preferred method of execution
- notebooks/main_concise.ipynb
  - Shortened version of standalone executable
  - Depends on custom modules
- main.py

*Note: All files require proper directory structure to work properly*

# 


