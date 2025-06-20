

# Data Pipeline Planning for Sensor Foundation Models

This document outlines the planning for data pipelines designed for training sensor foundation models. It covers dataset identification, unification, tool selection, and other key considerations.

# 1\. Dataset Identification

The first step is to identify all existing datasets that could be used to train sensor foundation models.

Here are the currently identified datasets:

| Dataset | Areas | Sandbox CNS | Label CNS |
| :---- | :---- | :---- | :---- |
| Tier-2 PROD | Activity, Mental |  |  |
| Wear-ME | Metabolic, Sleep, Mental |  |  |
| CGM | Metabolic |  |  |
| DWB | Sleep, Mental |  |  |
| Kereru | Sleep |  |  |
| Snowburn | Activity |  |  |
| PH-LLM | Sleep, Activity |  |  |
| Fitbit Pregnancy  | Women, Mental  |  |  |
| COVID |  |  |  |

# 2\. Unified Dataset Construction

To train effective foundation models, it's crucial to construct a unified dataset that integrates data from multiple sources.

Key considerations for unification include:

* **Data Filtering**  
  * Label selection   
  * Demographics selection   
  * Missing selection   
  * Subject number selection    
* **Centralized Data Meta Data Class**   
  * Sensor data location   
  * Label location   
  * Label types   
* **Label Processing Class per Dataset**  
  * Metabolic  
  * Wear-ME  
  * DWB   
  * …  
* **Centralized Sensor Processing Class**   
  * HR  
  * STEP  
  * …  
* **Centralized Caption Generation Class**   
  * Dataset 

We do not have to necessarily use \`Flume\` on all these datasets and tasks.   

