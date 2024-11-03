# Code completion model evaluation report

## Introduction
In this project, I aimed to generate a dataset of code completion examples using models from the open-source community, specifically `tiny_starcoder`, `codellama`, and `big_starcoder`. The dataset was derived from my LSTM price prediction project, which can be found [here](https://github.com/njaric03/crypto-lstm-predictor). The models were tested to evaluate their effectiveness in completing code snippets based on user cursor position.

## Dataset generation
I selected Python files from my repository and implemented a script to split the code into three segments: prefix (code before the cursor), middle (the code that is missing), and suffix (code after the cursor). The splitting was done randomly, resulting in a diverse set of examples suitable for evaluating code completion.

## Model selection and evaluation
After testing various models, I ultimately selected `codellama` for my analysis. To enhance performance, I configured the model to load in 4-bit mode, allowing me to run it on my local GPU with CUDA support. 

## Labeling and analysis
I adopted a labeling strategy that categorized completions into three labels: correct, partial, and incorrect. This choice was made because the resulting dataset produced no exact matches, but a considerable number of very close matches, which I deemed correct. By allowing for partial correctness, I aimed to capture the nuanced performance of the models better.

## Metric computation
I computed the following metrics to evaluate the quality of the code completions:
- **Exact Match**
- **chrF**
- **Token Overlap**
- **Levenshtein Distance**
- **Keyword Preservation**

I selected these metrics based on their relevance to the task and their popularity in the evaluation of code completion models. 
Through this project, I learned a lot about the metrics I chose, particularly finding the keyword preservation metric interesting, which turned out to be the most telling in my project.

## Results
The results of my evaluations, including analyses and visualizations of the metric distributions, are documented in the provided Jupyter notebook. These visualizations summarize model performance across the metrics I computed, which helped me interpret how well each metric performed relative to the others.

## Thought process and learnings
Throughout this project, I gained valuable insights into the complexities of code completion tasks. Testing multiple models provided a clear perspective on their varying performance levels. For instance, I found that `codellama` outperformed `tiny_starcoder` in generating coherent completions, especially when the context provided was more extensive. 

However, I also faced challenges, such as managing the long training times required for these complex models when running locally. This experience underscored the importance of optimizing model configurations, particularly the memory usage when working with larger models.

In exploring the chosen metrics, I deepened my understanding of their implications for evaluating code quality. Moving forward, I am eager to explore additional models and metrics that could further refine the evaluation process and enhance code generation capabilities.
