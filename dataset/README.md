#  EasyTS: The Express Lane to Long Time Series Forecasting

We introduce EasyTS, a compre-
hensive toolkit engineered to streamline the complex procedures of data collection, analysis, and model creation.  





## Dataset

EasyTS currently encompasses multiple datasets from six different domains around the world, 
with specific details provided in Figure 2. EasyTS introduces four novel opensource datasets related to electrical energy: FOOD, MANU,
PHAR, and OFFICE (Red indicates).

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="../pic/Dataset.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Figure 2: Datasets included in EasyTS</div>
</center>



### Electricity:
This dataset contains multiple power datasets, and the following is a list:

#### 1.PHAR (Pharmaceutical industry)
This dataset contains data on the maximum power demand (MD) for Pharmaceutical industries in Tianjin. The data is collected from the electricity summary table.

#### 2.FOOD (Food industry)
This dataset contains data on the maximum power demand (MD) for Food industries in Tianjin. The data is collected from the electricity summary table.

#### 3.MANU (Manufacturing industry)
This dataset contains data on the maximum power demand (MD) for Manufacturing industries in Tianjin. The data is collected from the electricity summary table.

#### 4.OFFICE (Office building)
This dataset contains data on the maximum power demand (MD) for Office buildings in Tianjin. The data is collected from the electricity summary table.

#### 5.ETT (Electricity Transformer Temperature)
The raw dataset is in https://github.com/zhouhaoyi/ETDataset .The ETT is a crucial indicator in the electric power long-term deployment. The dataset contains 2-year data from two separated counties in China and are separated as {ETTh1, ETTh2} for 1-hourlevel and ETTm1 for 15-minute-level. Each data point consists of the target value ”oil temperature” and 6 power load features.

#### 6.ECL (Electricity Consuming Load)

The raw dataset is in https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014. It is the electricity consumption in kWh was recorded every 15 minutes from 2011 to 2014. The dataset contains electircity consumption of 321 clients from 2012 to 2014. The data of it can reflect hourly consumption.

### Traffic
#### 1.traffic
The raw data is in http://pems.dot.ca.gov. The data in this repo is a collection of 48 months (2015-2016) hourly data from the California Department of Transportation. The data describes the road occupancy rates (between 0 and 1) measured by different sensors on San Francisco Bay area freeways.



### Cloud
#### 1.function
The raw data is in https://github.com/Azure/AzurePublicDataset. The dataset contains part of the workload of Microsoft's Azure Functions offering, collected in July of 2019. We selected one of the functions and performed some preprocessing to test the predictive performance of the temporal model in the cloud edge network environment.

#### 2.APP
The raw data is in https://github.com/ant-research/Pyraformer. This dataset was collected at Ant Group3. It consists of hourly maximum traffic flow for 128 systems deployed on 16 logic data centers, resulting in 1083 different time series in total. We recombine the data into two separate datasets as {app_a,app_b} to better fit the cloud-edge scenario. Each dataset is composed of multiple traffic flow.