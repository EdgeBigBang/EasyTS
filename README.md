#  EasyTS: The Express Lane to Long Time Series Forecasting

We introduce EasyTS, a comprehensive toolkit engineered to streamline the complex procedures of data collection, analysis, and model creation.  

[//]: # (### Paper)

[//]: # ([Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks]&#40;https://arxiv.org/abs/1703.07015&#41;)

# Overview

The overall architecture of this toolkit is illustrated in Figure 1.
EasyTS is structured into three progressively advanced lev-
els. Initially, it offers a one-stop solution for datasets, allow-
ing users to easily download and import richly-scenarioed
time series datasets with a single click through dataloader.
Subsequently, the toolkit embeds a variety of preprocessing
and convenient visualization analysis tools to aid researchers
in feature extraction and analysis. Building on this, an intu-
itive model building and validation interface is implemented
for rapid model development and assessment. In this stage,
EasyTS provides diverse evaluation metrics and benchmark
models to ensure comprehensive and in-depth model evalu-
ation.

<div style="text-align: center;">
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="pic/EasyTS_overview.png">
    <br>
    <div style="color:orange; text-align: center; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Figure 1: An overview of EasyTS</div>
</div>


## Rich One-stop Datasets Service

EasyTS currently encompasses multiple datasets from six different domains around the world, 
with specific details provided in Figure 2. EasyTS introduces four novel opensource datasets related to electrical energy: FOOD, MANU,
PHAR, and OFFICE (Red indicates).

<div style="text-align: center;">
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="pic/Dataset.png">
    <br>
    <div style="color:orange; text-align: center; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Figure 2: Datasets included in EasyTS</div>
</div>

For detailed information about the dataset, please refer to the readme file in the "dataset" folder.

## Convenient Data Visualization Component

EasyTS extends a versatile array of analytical tools, elegantly illustrated in Figure 3.

<div style="text-align: center;">
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="pic/Tools.png">
    <br>
    <div style="color:orange; text-align: center; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Figure 3: Various analytical tools provided by EasyTS</div>
</div>

## Convenient Model Verification Interface
Based on the two aspects mentioned above, scholars can swiftly design their own models within the toolkit's specified modules. EasyTS provides diverse time-series evaluation metrics, such as MSE, MAE, DDTW, CORR, RMSE, to objectively assess the sophistication of the models, with the scoring results displayed in Figure 4.

<div style="text-align: center;">
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="pic/Metrics.png">
    <br>
    <div style="text-align: center; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Figure 4: Diverse time-series evaluation metrics</div>
</div>