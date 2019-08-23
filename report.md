# Machine Learning Engineer Nanodegree
## Capstone Project
Chitrang Patel  
August 22, 2019

## I. Definition

### Project Overview
Technical analysis is a trading discipline employed to evaluate investments and identify trading opportunities by analyzing statistical trends gathered from trading activity, such as price movement and volume. Recent advances in Deep Learning for financial trading have shown to outperform human traders. It is incredibly valuable for ameture investors!

This project uses Recurrant Neural Nets (RNNs), particularly the Long Short Term Memory (LSTM) architechture, to predict stock prices. This technique is actively being studied and used in time series forecasting situations and also to real-world trading platforms. This project will use the Keras library to build a LSTM model to predict the adjusted closing value of stocks using historical data.

### Problem Statement

The main task for this project is to accurately predict the future adjusted closing value of a given stock for the next day given the previous day's values. The stocks that I used for this project are AAPL, MSFT, AMZN, GOOGL and NVDA.
The approach to this problem was as follows:
- Download the data using the Yahoo Historical package from pip. The data used for this project was from 2009-01-01 to 2019-01-01.
- Extract the features used in training the model and the target that we want to predict.
- Split the features and target into training and testing sets.
- Make the benchmark Linear Regression Model.
- Use the model to fit the training dataset.
- Use the trained model to predict the target values of the test dataset. 
- Evaluate the performance of this model by comparing it to the actual target values.
- Make the RNN model using the LSTM architechture.
- Use this model to fit the training dataset and learn from it.
- Use the trained model to predict the target values of the test dataset. 
- Evaluate the performance of this model by comparing it to the actual target values.

We want to then compare performance between the Linear Regression model and the RNN model.

### Metrics

Since this is a regression type of a problem, I use the R-square score and the root mean squared error as performance metrics.
`R-square` is a statistical measure of how much the variation in the target variable can be explained by the dependent variables (features) in a regression model.
`Root mean squared error` is the square root of the mean of the squares of the difference between the predicted values from the model and the actual (true) values.
Both these metrics are used heavily for regression problems in the literature.

## II. Analysis
### Data Exploration

This project explores the data for the following companies `AAPL` (Apple), `GOOGL` (Google), `AMZN` (Amazon), `NVDA` (Nvidea), and `MSFT` (Microsoft). The data for the analysis was extracted using the PyPI package `yahoo historical` from `2009-01-01` to `2019-01-01`. The data consisted of `Open`, `High`, `Low`, `Close`, `Adj Close`, and `Volume` for each day. The goal of this project is to predict the `Adj Close` price for the following day using the information in the past. Over the duration of 10 years, there were data for 2516 days since the market is open from Monday to Friday. The dataset for each ticker is summarized below in their respective tables -- `Count`, `Mean`, `Std deviation`, `Min`, and `Max` values. As clearly shown, the `Volume` values are very large compared to the other values. In order to run the models successfully, we need to scale the data to the same level. There were no missing values in the dataset.  

<table>
<caption>AAPL</caption>
<tr><th></th> <th>Open</th> <th>High</th> <th>Low</th> <th>Close</th> <th>Adj Close</th> <th>Volume</th></tr>
<tr><td>count</td><td style="text-align: right;">2516     </td><td style="text-align: right;">2516     </td><td style="text-align: right;">2516     </td><td style="text-align: right;">2516     </td><td style="text-align: right;">2516      </td><td style="text-align: right;">2516          </td></tr>
<tr><td>mean </td><td style="text-align: right;">  91.5972</td><td style="text-align: right;">  92.4069</td><td style="text-align: right;">  90.7341</td><td style="text-align: right;">  91.5873</td><td style="text-align: right;">  85.4103 </td><td style="text-align: right;">   8.6325e+07 </td></tr>
<tr><td>std  </td><td style="text-align: right;">  50.2748</td><td style="text-align: right;">  50.6859</td><td style="text-align: right;">  49.8551</td><td style="text-align: right;">  50.2726</td><td style="text-align: right;">  50.3187 </td><td style="text-align: right;">   6.1923e+07 </td></tr>
<tr><td>min  </td><td style="text-align: right;">  11.3414</td><td style="text-align: right;">  11.7143</td><td style="text-align: right;">  11.1714</td><td style="text-align: right;">  11.1714</td><td style="text-align: right;">   9.74945</td><td style="text-align: right;">   1.14759e+07</td></tr>
<tr><td>max  </td><td style="text-align: right;"> 230.78  </td><td style="text-align: right;"> 233.47  </td><td style="text-align: right;"> 229.78  </td><td style="text-align: right;"> 232.07  </td><td style="text-align: right;"> 228.524  </td><td style="text-align: right;">   4.7025e+08 </td></tr>
</table>


<table>
<caption>GOOGL</caption>
<tr><th></th> <th>Open</th> <th>High</th> <th>Low</th> <th>Close</th> <th>Adj Close</th> <th>Volume</th></tr>
<tr><td>count</td><td style="text-align: right;">2516    </td><td style="text-align: right;">2516    </td><td style="text-align: right;">2516    </td><td style="text-align: right;">2516    </td><td style="text-align: right;">2516    </td><td style="text-align: right;">  2516          </td></tr>
<tr><td>mean </td><td style="text-align: right;"> 554.972</td><td style="text-align: right;"> 559.659</td><td style="text-align: right;"> 549.823</td><td style="text-align: right;"> 554.842</td><td style="text-align: right;"> 554.842</td><td style="text-align: right;">     3.9364e+06 </td></tr>
<tr><td>std  </td><td style="text-align: right;"> 296.359</td><td style="text-align: right;"> 298.913</td><td style="text-align: right;"> 293.517</td><td style="text-align: right;"> 296.245</td><td style="text-align: right;"> 296.245</td><td style="text-align: right;">     2.99744e+06</td></tr>
<tr><td>min  </td><td style="text-align: right;"> 144.319</td><td style="text-align: right;"> 149.9  </td><td style="text-align: right;"> 141.517</td><td style="text-align: right;"> 141.517</td><td style="text-align: right;"> 141.517</td><td style="text-align: right;">520600          </td></tr>
<tr><td>max  </td><td style="text-align: right;">1289.12 </td><td style="text-align: right;">1291.44 </td><td style="text-align: right;">1263    </td><td style="text-align: right;">1285.5  </td><td style="text-align: right;">1285.5  </td><td style="text-align: right;">     2.96199e+07</td></tr>
</table>
</div>


<table>
<caption>AMZN</caption>
<tr><th></th> <th>Open</th> <th>High</th> <th>Low</th> <th>Close</th> <th>Adj Close</th> <th>Volume</th></tr>
<tr><td>count</td><td style="text-align: right;">2516    </td><td style="text-align: right;">2516    </td><td style="text-align: right;">2516    </td><td style="text-align: right;">2516    </td><td style="text-align: right;">2516    </td><td style="text-align: right;">  2516          </td></tr>
<tr><td>mean </td><td style="text-align: right;"> 505.953</td><td style="text-align: right;"> 511.113</td><td style="text-align: right;"> 499.909</td><td style="text-align: right;"> 505.746</td><td style="text-align: right;"> 505.746</td><td style="text-align: right;">     4.99449e+06</td></tr>
<tr><td>std  </td><td style="text-align: right;"> 467.136</td><td style="text-align: right;"> 471.579</td><td style="text-align: right;"> 461.101</td><td style="text-align: right;"> 466.377</td><td style="text-align: right;"> 466.377</td><td style="text-align: right;">     3.44577e+06</td></tr>
<tr><td>min  </td><td style="text-align: right;">  48.56 </td><td style="text-align: right;">  50.1  </td><td style="text-align: right;">  47.63 </td><td style="text-align: right;">  48.44 </td><td style="text-align: right;">  48.44 </td><td style="text-align: right;">984400          </td></tr>
<tr><td>max  </td><td style="text-align: right;">2038.11 </td><td style="text-align: right;">2050.5  </td><td style="text-align: right;">2013    </td><td style="text-align: right;">2039.51 </td><td style="text-align: right;">2039.51 </td><td style="text-align: right;">     5.83058e+07</td></tr>
</table>

<table>
<caption>NVDA</caption>
<tr><th></th> <th>Open</th> <th>High</th> <th>Low</th> <th>Close</th> <th>Adj Close</th> <th>Volume</th></tr>
<tr><td>count</td><td style="text-align: right;">2516     </td><td style="text-align: right;">2516     </td><td style="text-align: right;">2516     </td><td style="text-align: right;">2516     </td><td style="text-align: right;">2516      </td><td style="text-align: right;">2516          </td></tr>
<tr><td>mean </td><td style="text-align: right;">  54.7694</td><td style="text-align: right;">  55.5706</td><td style="text-align: right;">  53.8663</td><td style="text-align: right;">  54.7438</td><td style="text-align: right;">  53.8381 </td><td style="text-align: right;">   1.38323e+07</td></tr>
<tr><td>std  </td><td style="text-align: right;">  73.8333</td><td style="text-align: right;">  74.8638</td><td style="text-align: right;">  72.5991</td><td style="text-align: right;">  73.7494</td><td style="text-align: right;">  73.7429 </td><td style="text-align: right;">   9.06445e+06</td></tr>
<tr><td>min  </td><td style="text-align: right;">   7.21  </td><td style="text-align: right;">   7.47  </td><td style="text-align: right;">   7.08  </td><td style="text-align: right;">   7.21  </td><td style="text-align: right;">   6.65138</td><td style="text-align: right;">   1.1411e+06 </td></tr>
<tr><td>max  </td><td style="text-align: right;"> 289.32  </td><td style="text-align: right;"> 292.76  </td><td style="text-align: right;"> 285.58  </td><td style="text-align: right;"> 289.36  </td><td style="text-align: right;"> 288.444  </td><td style="text-align: right;">   9.23232e+07</td></tr>
</table>


<table>
<caption>MSFT</caption>
<tr><th></th> <th>Open</th> <th>High</th> <th>Low</th> <th>Close</th> <th>Adj Close</th> <th>Volume</th></tr>
<tr><td>count</td><td style="text-align: right;">2516     </td><td style="text-align: right;">2516     </td><td style="text-align: right;">2516     </td><td style="text-align: right;">2516     </td><td style="text-align: right;">2516     </td><td style="text-align: right;">2516          </td></tr>
<tr><td>mean </td><td style="text-align: right;">  45.5529</td><td style="text-align: right;">  45.9534</td><td style="text-align: right;">  45.1356</td><td style="text-align: right;">  45.5646</td><td style="text-align: right;">  41.4448</td><td style="text-align: right;">   4.37174e+07</td></tr>
<tr><td>std  </td><td style="text-align: right;">  23.9477</td><td style="text-align: right;">  24.1402</td><td style="text-align: right;">  23.6991</td><td style="text-align: right;">  23.9257</td><td style="text-align: right;">  24.919 </td><td style="text-align: right;">   2.47416e+07</td></tr>
<tr><td>min  </td><td style="text-align: right;">  15.2   </td><td style="text-align: right;">  15.62  </td><td style="text-align: right;">  14.87  </td><td style="text-align: right;">  15.15  </td><td style="text-align: right;">  11.7711</td><td style="text-align: right;">   7.4256e+06 </td></tr>
<tr><td>max  </td><td style="text-align: right;"> 115.42  </td><td style="text-align: right;"> 116.18  </td><td style="text-align: right;"> 114.93  </td><td style="text-align: right;"> 115.61  </td><td style="text-align: right;"> 113.821 </td><td style="text-align: right;">   3.19318e+08</td></tr>
</table>

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
